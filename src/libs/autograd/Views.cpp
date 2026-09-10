#include "Shape.h"
#include "Storage.h"
#include "Strides.h"
#include "Variable.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

// The gradients a transformer needs beyond arithmetic: it splits heads, swaps axes and rotates
// channels, and the backward pass has to undo those movements as surely as it undoes a
// multiplication.
//
// All three are linear maps that MOVE numbers without combining them, and for such a map the
// backward is the same map run in reverse. Nothing is differentiated; the only thing that can go
// wrong is applying the inverse the wrong way round, which is why every test uses a non-square
// shape.
namespace veda::autograd
{

using core::Shape;
using core::Tensor;

namespace
{
Variable make_unary(const Variable& x, Tensor value, std::function<void(const Tensor&)> backward)
{
    if (!x.requires_grad())
    {
        return Variable::leaf(std::move(value), false);
    }
    auto node = std::make_shared<Node>(std::move(value));
    node->requires_grad = true;
    node->parents = {x.node()};
    node->backward = std::move(backward);
    return Variable(std::move(node));
}

void accumulate_into(Tensor& destination, const Tensor& addition)
{
    for (size_t i = 0; i < destination.numel(); ++i)
    {
        destination.data()[i] += addition.data()[i];
    }
}
} // namespace

Variable reshape(const Variable& x, const Shape& shape)
{
    // Nothing moves: the same buffer, read by a different rule (E1.S2.T5).
    Tensor value = x.value().reshape(shape);
    auto input = x.node();
    const Shape original = x.value().shape();

    return make_unary(x, std::move(value), [input, original](const Tensor& g) {
        accumulate_into(input->grad, g.reshape(original));
    });
}

Variable transpose(const Variable& x, size_t a, size_t b)
{
    // The forward path leaves a transposed tensor as a view, and that costs nothing (AD2). Here the
    // value is MATERIALISED contiguous, and the reason is not elegance: several backward passes on
    // this tape index their input as a flat buffer, which is correct only for a contiguous tensor.
    // Handing them a strided view would give wrong gradients silently — the worst failure mode
    // there is (risk R8). A copy per transpose is what that guarantee costs, and E17 may revisit it
    // with a profile in hand.
    Tensor value = core::contiguous(x.value().transpose(a, b));
    auto input = x.node();

    return make_unary(x, std::move(value), [input, a, b](const Tensor& g) {
        // The same two axes swapped again — which is what makes transpose its own inverse.
        accumulate_into(input->grad, core::contiguous(g.transpose(a, b)));
    });
}

Variable rope(const Variable& x, const Tensor& cosines, const Tensor& sines)
{
    // The tables are passed as plain tensors rather than a model::RopeTables so that the tape keeps
    // depending on core alone; the caller builds them with model::rope_tables.
    const Shape& shape = x.value().shape();
    if (shape.rank() < 2)
    {
        throw std::invalid_argument("autograd::rope: expected [.., T, Dh], got " +
                                    shape.to_string());
    }
    const size_t head_dim = shape[shape.rank() - 1];
    const size_t positions = shape[shape.rank() - 2];
    if (cosines.shape() != Shape({positions, head_dim}) ||
        sines.shape() != Shape({positions, head_dim}))
    {
        throw std::invalid_argument("autograd::rope: tables do not match (" +
                                    std::to_string(positions) + ", " +
                                    std::to_string(head_dim) + ")");
    }
    if (!x.value().is_contiguous())
    {
        throw std::invalid_argument("autograd::rope: expects a contiguous input");
    }

    // out[i]   = x[i]cos - x[i+p]sin
    // out[i+p] = x[i+p]cos + x[i]sin        the half-split pairing of E8
    Tensor value{shape};
    {
        const size_t pairs = head_dim / 2;
        const size_t lanes = head_dim == 0 || positions == 0 ? 0
                                                             : x.value().numel() /
                                                                   (positions * head_dim);
        const float* source = x.value().data();
        float* destination = value.data();
        for (size_t lane = 0; lane < lanes; ++lane)
        {
            const size_t base = lane * positions * head_dim;
            for (size_t p = 0; p < positions; ++p)
            {
                const size_t row = base + p * head_dim;
                for (size_t i = 0; i < pairs; ++i)
                {
                    const float cosine = cosines.data()[p * head_dim + i];
                    const float sine = sines.data()[p * head_dim + i];
                    const float first = source[row + i];
                    const float second = source[row + i + pairs];
                    destination[row + i] = first * cosine - second * sine;
                    destination[row + i + pairs] = second * cosine + first * sine;
                }
            }
        }
    }

    auto input = x.node();

    return make_unary(x, std::move(value), [input, cosines, sines](const Tensor& g) {
        // Per pair the forward is the matrix [[c, -s], [s, c]]. A rotation is ORTHOGONAL: its
        // transpose is its inverse, which is rotation by -theta. So the backward is the same
        // routine with the sine's sign flipped.
        //
        // Nothing flows to the tables. They are fixed functions of position, not parameters — which
        // is the whole point of RoPE and the reason it needs no training.
        const Shape& shape = g.shape();
        const size_t head_dim = shape[shape.rank() - 1];
        const size_t positions = shape[shape.rank() - 2];
        const size_t pairs = head_dim / 2;
        const size_t lanes = g.numel() / (positions * head_dim);

        Tensor grad{shape};
        const float* source = g.data();
        float* destination = grad.data();

        for (size_t lane = 0; lane < lanes; ++lane)
        {
            const size_t base = lane * positions * head_dim;
            for (size_t p = 0; p < positions; ++p)
            {
                const size_t row = base + p * head_dim;
                for (size_t i = 0; i < pairs; ++i)
                {
                    const float cosine = cosines.data()[p * head_dim + i];
                    const float sine = sines.data()[p * head_dim + i];
                    const float first = source[row + i];
                    const float second = source[row + i + pairs];

                    destination[row + i] = first * cosine + second * sine;
                    destination[row + i + pairs] = second * cosine - first * sine;
                }
            }
        }

        accumulate_into(input->grad, grad);
    });
}

} // namespace veda::autograd
