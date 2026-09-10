#include "Activations.h"
#include "Elementwise.h"
#include "Normalize.h"
#include "Shape.h"
#include "Softmax.h"
#include "Storage.h"
#include "Variable.h"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace veda::autograd
{

using core::Shape;
using core::Tensor;

namespace
{

// Builds a node for a unary op, or a bare leaf when nothing needs a gradient.
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

Variable silu(const Variable& x)
{
    Tensor value = ops::silu(x.value());
    auto input = x.node();

    return make_unary(x, std::move(value), [input](const Tensor& g) {
        // d/dx [x . sigma(x)] = sigma(x) . (1 + x(1 - sigma(x)))
        Tensor grad{input->value.shape()};
        for (size_t i = 0; i < grad.numel(); ++i)
        {
            const float v = input->value.data()[i];
            const float s = 1.0f / (1.0f + std::exp(-v));
            grad.data()[i] = g.data()[i] * s * (1.0f + v * (1.0f - s));
        }
        accumulate_into(input->grad, grad);
    });
}

Variable gelu(const Variable& x)
{
    Tensor value = ops::gelu(x.value());
    auto input = x.node();

    return make_unary(x, std::move(value), [input](const Tensor& g) {
        // 0.5(1 + tanh u) + 0.5 x (1 - tanh^2 u) c (1 + 3 . 0.044715 x^2),  u = c(x + 0.044715x^3)
        constexpr float c = 0.7978845608028654f;
        constexpr float cubic = 0.044715f;

        Tensor grad{input->value.shape()};
        for (size_t i = 0; i < grad.numel(); ++i)
        {
            const float v = input->value.data()[i];
            const float u = c * (v + cubic * v * v * v);
            const float t = std::tanh(u);
            const float derivative =
                0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * c * (1.0f + 3.0f * cubic * v * v);
            grad.data()[i] = g.data()[i] * derivative;
        }
        accumulate_into(input->grad, grad);
    });
}

Variable softmax(const Variable& x, size_t dim)
{
    Tensor value = ops::softmax(x.value(), dim);
    auto input = x.node();
    const Tensor output = value;   // the backward needs y, not x

    return make_unary(x, std::move(value), [input, output, dim](const Tensor& g) {
        // The Jacobian dy_i/dx_j = y_i(delta_ij - y_j) is DENSE — 23 billion entries at
        // V = 151,936, which is not slow but impossible. The gradient never needs it, only its
        // product with G:
        //
        //     dL/dx_i = y_i . (G_i - sum_j G_j y_j)
        //
        // One weighted sum per lane, then an element-wise subtract and multiply. Linear, not
        // quadratic.
        const Shape& shape = output.shape();
        const size_t length = shape[dim];
        const size_t lanes = shape.size() == 0 ? 0 : shape.size() / length;

        Tensor grad{shape};
        std::vector<size_t> index(shape.rank(), 0);

        for (size_t lane = 0; lane < lanes; ++lane)
        {
            auto offset_of = [&](size_t position) {
                size_t flat = 0;
                for (size_t k = 0; k < shape.rank(); ++k)
                {
                    const size_t coordinate = k == dim ? position : index[k];
                    flat = flat * shape[k] + coordinate;
                }
                return flat;
            };

            float weighted = 0.0f;
            for (size_t i = 0; i < length; ++i)
            {
                const size_t at = offset_of(i);
                weighted += g.data()[at] * output.data()[at];
            }
            for (size_t i = 0; i < length; ++i)
            {
                const size_t at = offset_of(i);
                grad.data()[at] = output.data()[at] * (g.data()[at] - weighted);
            }

            for (size_t k = shape.rank(); k-- > 0;)
            {
                if (k == dim)
                {
                    continue;
                }
                if (++index[k] < shape[k])
                {
                    break;
                }
                index[k] = 0;
            }
        }

        accumulate_into(input->grad, grad);
    });
}

Variable rms_normalize(const Variable& x, float eps)
{
    Tensor value = ops::rms_normalize(x.value(), eps);
    auto input = x.node();

    return make_unary(x, std::move(value), [input, eps](const Tensor& g) {
        // Normalisation ties a lane together: a change in one element moves every other element's
        // output, through r. So the gradient has a second, coupling term:
        //
        //     dL/dx_i = G_i/r - x_i/(n r^3) . sum_j G_j x_j
        //
        // Dropping it gives a gradient that looks plausible and trains badly.
        const Tensor& x_value = input->value;
        const Shape& shape = x_value.shape();
        if (shape.rank() == 0 || shape.size() == 0)
        {
            return;
        }

        const size_t length = shape[shape.rank() - 1];
        const size_t lanes = shape.size() / length;

        Tensor grad{shape};
        for (size_t lane = 0; lane < lanes; ++lane)
        {
            const size_t base = lane * length;

            float squares = 0.0f;
            float weighted = 0.0f;
            for (size_t i = 0; i < length; ++i)
            {
                const float v = x_value.data()[base + i];
                squares += v * v;
                weighted += g.data()[base + i] * v;
            }

            const float r = std::sqrt(squares / static_cast<float>(length) + eps);
            const float coupling = weighted / (static_cast<float>(length) * r * r * r);

            for (size_t i = 0; i < length; ++i)
            {
                grad.data()[base + i] =
                    g.data()[base + i] / r - x_value.data()[base + i] * coupling;
            }
        }

        accumulate_into(input->grad, grad);
    });
}

Variable embedding(const Variable& table, const std::vector<int64_t>& ids, const Shape& id_shape)
{
    if (table.value().rank() != 2)
    {
        throw std::invalid_argument("autograd::embedding: the table must be rank 2 [V, D], got " +
                                    table.value().shape().to_string());
    }
    if (ids.size() != id_shape.size())
    {
        throw std::invalid_argument("autograd::embedding: " + std::to_string(ids.size()) +
                                    " ids for a shape " + id_shape.to_string());
    }

    const size_t vocabulary = table.value().shape()[0];
    const size_t hidden = table.value().shape()[1];

    std::vector<size_t> dims = id_shape.dims();
    dims.push_back(hidden);
    Tensor value{Shape(std::move(dims))};

    for (size_t p = 0; p < ids.size(); ++p)
    {
        const int64_t id = ids[p];
        if (id < 0 || static_cast<size_t>(id) >= vocabulary)
        {
            throw std::out_of_range("autograd::embedding: token id " + std::to_string(id) +
                                    " is outside a vocabulary of " + std::to_string(vocabulary));
        }
        for (size_t d = 0; d < hidden; ++d)
        {
            value.data()[p * hidden + d] = table.value().at({static_cast<size_t>(id), d});
        }
    }

    auto input = table.node();
    return make_unary(table, std::move(value), [input, ids, hidden](const Tensor& g) {
        // A scatter-add, not a matmul: the forward copied row `id`, so the backward adds G's row
        // back into that row's gradient. Repeated ids ACCUMULATE — a token appearing three times
        // contributes three times.
        //
        // Nothing flows to the ids themselves: they are indices, and there is nothing to
        // differentiate with respect to.
        for (size_t p = 0; p < ids.size(); ++p)
        {
            const size_t row = static_cast<size_t>(ids[p]);
            for (size_t d = 0; d < hidden; ++d)
            {
                input->grad.data()[row * hidden + d] += g.data()[p * hidden + d];
            }
        }
    });
}

} // namespace veda::autograd
