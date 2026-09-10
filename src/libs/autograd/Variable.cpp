#include "Variable.h"

#include "Broadcast.h"
#include "Elementwise.h"
#include "Matmul.h"
#include "Shape.h"
#include "Storage.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace veda::autograd
{

using core::Shape;
using core::Tensor;

namespace
{

Tensor zeros_like(const Tensor& t)
{
    return Tensor{t.shape()};
}

void accumulate(Tensor& destination, const Tensor& addition)
{
    if (destination.shape() != addition.shape())
    {
        throw std::runtime_error("autograd: a gradient of shape " + addition.shape().to_string() +
                                 " does not fit a value of shape " + destination.shape().to_string());
    }
    for (size_t i = 0; i < destination.numel(); ++i)
    {
        destination.data()[i] += addition.data()[i];
    }
}

} // namespace

namespace detail
{

// The backward of a broadcast is a sum: if [D] was broadcast against [B, T, D], each of its
// elements contributed to B*T outputs, so its gradient is the sum over those axes. Getting this
// wrong produces a gradient of the wrong shape, which accumulate() catches.
Tensor unbroadcast(const Tensor& gradient, const Shape& target)
{
    if (gradient.shape() == target)
    {
        return gradient;
    }

    Tensor reduced{target};
    const size_t extra = gradient.rank() - target.rank();

    std::vector<size_t> index(gradient.rank(), 0);
    for (size_t i = 0; i < gradient.numel(); ++i)
    {
        // The source element's coordinates, with the leading axes dropped and any axis of extent 1
        // in the target collapsed to 0.
        size_t offset = 0;
        for (size_t k = 0; k < target.rank(); ++k)
        {
            const size_t coordinate = target[k] == 1 ? 0 : index[k + extra];
            offset = offset * target[k] + coordinate;
        }

        size_t flat = gradient.offset();
        for (size_t k = 0; k < gradient.rank(); ++k)
        {
            flat += index[k] * gradient.strides()[k];
        }
        reduced.data()[offset] += gradient.storage()->data()[flat];

        for (size_t k = gradient.rank(); k-- > 0;)
        {
            if (++index[k] < gradient.shape()[k])
            {
                break;
            }
            index[k] = 0;
        }
    }
    return reduced;
}

} // namespace detail

namespace
{
using detail::unbroadcast;
}

Node::Node(Tensor v) : value(std::move(v)), grad(zeros_like(value)) {}

Variable Variable::leaf(Tensor value, bool requires_grad)
{
    auto node = std::make_shared<Node>(std::move(value));
    node->requires_grad = requires_grad;
    return Variable(std::move(node));
}

void Variable::zero_grad()
{
    std::unordered_set<Node*> seen;
    std::vector<Node*> pending{node_.get()};

    while (!pending.empty())
    {
        Node* node = pending.back();
        pending.pop_back();
        if (!seen.insert(node).second)
        {
            continue;
        }
        node->grad = zeros_like(node->value);
        for (const auto& parent : node->parents)
        {
            pending.push_back(parent.get());
        }
    }
}

void Variable::backward()
{
    if (node_->value.numel() != 1)
    {
        throw std::runtime_error("autograd: backward() starts at a scalar, got " +
                                 node_->value.shape().to_string());
    }

    // Reverse topological order: a node is applied only once every consumer of it has contributed.
    std::vector<Node*> order;
    std::unordered_set<Node*> seen;

    std::function<void(Node*)> visit = [&](Node* node) {
        if (!seen.insert(node).second)
        {
            return;
        }
        for (const auto& parent : node->parents)
        {
            visit(parent.get());
        }
        order.push_back(node);
    };
    visit(node_.get());

    // Intermediate gradients are scratch space for ONE backward pass, so they start at zero every
    // time; leaves accumulate across passes, which is what makes gradient accumulation over
    // micro-batches work. Keeping an intermediate's gradient between passes would double its
    // contribution the second time — a second backward would give 12 where 8 is right.
    for (Node* node : order)
    {
        if (node->backward)
        {
            node->grad = core::Tensor{node->value.shape()};
        }
    }

    node_->grad.data()[0] = 1.0f;

    for (auto it = order.rbegin(); it != order.rend(); ++it)
    {
        if ((*it)->backward)
        {
            (*it)->backward((*it)->grad);
        }
    }
}

namespace
{

// Builds a node for a binary op, or a bare leaf when nothing needs a gradient.
Variable make(const Variable& a, const Variable& b, Tensor value,
              std::function<void(const Tensor&)> backward)
{
    if (!a.requires_grad() && !b.requires_grad())
    {
        return Variable::leaf(std::move(value), false);
    }

    auto node = std::make_shared<Node>(std::move(value));
    node->requires_grad = true;
    node->parents = {a.node(), b.node()};
    node->backward = std::move(backward);
    return Variable(std::move(node));
}

} // namespace

Variable add(const Variable& a, const Variable& b)
{
    Tensor value = ops::add(a.value(), b.value());

    auto left = a.node();
    auto right = b.node();
    const Shape left_shape = left->value.shape();
    const Shape right_shape = right->value.shape();

    return make(a, b, std::move(value), [left, right, left_shape, right_shape](const Tensor& g) {
        // d(a+b)/da = 1, d(a+b)/db = 1 — the gradient passes straight through, summed back over
        // whatever was broadcast.
        if (left->requires_grad)
        {
            accumulate(left->grad, unbroadcast(g, left_shape));
        }
        if (right->requires_grad)
        {
            accumulate(right->grad, unbroadcast(g, right_shape));
        }
    });
}

Variable mul(const Variable& a, const Variable& b)
{
    Tensor value = ops::mul(a.value(), b.value());

    auto left = a.node();
    auto right = b.node();
    const Shape left_shape = left->value.shape();
    const Shape right_shape = right->value.shape();

    return make(a, b, std::move(value), [left, right, left_shape, right_shape](const Tensor& g) {
        // d(ab)/da = b, d(ab)/db = a. When a and b are the same node both contributions accumulate
        // into it, which is why mul(x, x) gives 2x rather than x.
        if (left->requires_grad)
        {
            accumulate(left->grad, unbroadcast(ops::mul(g, right->value), left_shape));
        }
        if (right->requires_grad)
        {
            accumulate(right->grad, unbroadcast(ops::mul(g, left->value), right_shape));
        }
    });
}

Variable mul(const Variable& a, float scalar)
{
    Tensor value = ops::mul(a.value(), scalar);

    auto input = a.node();
    if (!a.requires_grad())
    {
        return Variable::leaf(std::move(value), false);
    }

    auto node = std::make_shared<Node>(std::move(value));
    node->requires_grad = true;
    node->parents = {input};
    node->backward = [input, scalar](const Tensor& g) {
        accumulate(input->grad, ops::mul(g, scalar));
    };
    return Variable(std::move(node));
}

Variable sum(const Variable& x)
{
    Tensor total{Shape({})};
    for (size_t i = 0; i < x.value().numel(); ++i)
    {
        total.data()[0] += x.value().data()[i];
    }

    auto input = x.node();
    if (!x.requires_grad())
    {
        return Variable::leaf(std::move(total), false);
    }

    auto node = std::make_shared<Node>(std::move(total));
    node->requires_grad = true;
    node->parents = {input};
    node->backward = [input](const Tensor& g) {
        // Every element contributed once, so every element's gradient is the scalar's.
        Tensor spread{input->value.shape()};
        for (size_t i = 0; i < spread.numel(); ++i)
        {
            spread.data()[i] = g.data()[0];
        }
        accumulate(input->grad, spread);
    };
    return Variable(std::move(node));
}

namespace
{

// Transposes the last two axes of a rank >= 2 tensor.
Tensor transpose_matrix(const Tensor& t)
{
    return t.transpose(t.rank() - 2, t.rank() - 1);
}

} // namespace

Variable matmul(const Variable& a, const Variable& b)
{
    Tensor value = ops::matmul(a.value(), b.value());

    auto left = a.node();
    auto right = b.node();
    const Shape left_shape = left->value.shape();
    const Shape right_shape = right->value.shape();

    return make(a, b, std::move(value), [left, right, left_shape, right_shape](const Tensor& g) {
        if (left->requires_grad)
        {
            // dL/dA[i,k] = sum_j G[i,j] . B[k,j]. G is [.., M, N] and B is [.., K, N]: the shared
            // index is the LAST of both, so this is matmul_nt with B as it stands — no transpose.
            accumulate(left->grad, unbroadcast(ops::matmul_nt(g, right->value), left_shape));
        }
        if (right->requires_grad)
        {
            // dL/dB = At . G, summed back over whatever batch axes B was broadcast across.
            accumulate(right->grad,
                       unbroadcast(ops::matmul(transpose_matrix(left->value), g), right_shape));
        }
    });
}

Variable matmul_nt(const Variable& a, const Variable& b)
{
    Tensor value = ops::matmul_nt(a.value(), b.value());

    auto left = a.node();
    auto right = b.node();
    const Shape left_shape = left->value.shape();
    const Shape right_shape = right->value.shape();

    return make(a, b, std::move(value), [left, right, left_shape, right_shape](const Tensor& g) {
        if (left->requires_grad)
        {
            // The mirror: the forward's _nt makes this one a plain matmul.
            accumulate(left->grad, unbroadcast(ops::matmul(g, right->value), left_shape));
        }
        if (right->requires_grad)
        {
            accumulate(right->grad,
                       unbroadcast(ops::matmul(transpose_matrix(g), left->value), right_shape));
        }
    });
}

} // namespace veda::autograd
