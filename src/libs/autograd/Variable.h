#ifndef VEDA_VARIABLE_H
#define VEDA_VARIABLE_H

#include "Shape.h"
#include "Tensor.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Reverse-mode automatic differentiation over the ops E2 built.
//
// Training needs the gradient of one scalar loss with respect to every parameter. Computing that
// one parameter at a time would take as many forward passes as there are parameters; reverse mode
// computes all of them in ONE backward pass costing about what a forward pass costs. That factor is
// why neural networks are trainable at all.
//
// This is an addition rather than a rewrite because of a decision taken in E2 for other reasons:
// ops are pure functions that never mutate their inputs (architecture §7), so a node can hold its
// inputs and be sure they are still what they were. An in-place add_ would have made every node a
// lie.
namespace veda::autograd
{

class Variable;

// A node of the tape: a value, its gradient, its inputs, and how to turn a gradient at the output
// into gradients at the inputs.
struct Node
{
    core::Tensor value;
    core::Tensor grad;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void(const core::Tensor& out_grad)> backward;
    bool requires_grad = false;

    explicit Node(core::Tensor v);
};

class Variable
{
public:
    static Variable leaf(core::Tensor value, bool requires_grad = true);
    explicit Variable(std::shared_ptr<Node> node) : node_(std::move(node)) {}

    const core::Tensor& value() const noexcept { return node_->value; }
    const core::Tensor& grad() const noexcept { return node_->grad; }
    bool requires_grad() const noexcept { return node_->requires_grad; }
    const std::shared_ptr<Node>& node() const noexcept { return node_; }

    // Seeds 1 at a scalar and walks the tape in reverse topological order — applying a node before
    // all of its consumers have contributed would propagate an incomplete gradient, which is the
    // classic bug and the reason any traversal will not do.
    //
    // Gradients ACCUMULATE, never assign: a value used twice contributes twice, so y = x + x gives
    // dL/dx = 2. Forgetting that produces a model that trains strangely rather than one that errors.
    void backward();

    void zero_grad();

private:
    std::shared_ptr<Node> node_;
};

// When no input requires a gradient the result is a leaf with no parents and no closure: inference
// pays nothing, which is what keeps veda::ops usable exactly as before.
Variable add(const Variable& a, const Variable& b);
Variable mul(const Variable& a, const Variable& b);
Variable mul(const Variable& a, float scalar);
Variable sum(const Variable& x);   // a scalar, so that there is something to differentiate

// C = A . B, with the gradients that follow from the chain rule:
//
//     dL/dA[i,k] = sum_j G[i,j] . B[k,j]  ->  matmul_nt(G, B)
//     dL/dB[k,j] = sum_i G[i,j] . A[i,k]  ->  matmul(transpose(A), G)
//
// Note the mirror: the backward of a plain matmul is written with matmul_nt, and the backward of
// matmul_nt is written with a plain matmul. The two variants are each other's transpose, which is
// the clearest argument that AD4's decision to build both was right.
//
// When a weight [out, in] meets activations [B, T, in] it was broadcast across B*T positions, so
// its gradient is summed over them. Forgetting that sum gives a gradient of the wrong shape;
// scaling it wrongly gives one of the right shape that is quietly B*T times too large — the first
// is caught by the shape assertion, the second only by finite differences.
Variable matmul(const Variable& a, const Variable& b);
Variable matmul_nt(const Variable& a, const Variable& b);

// The non-linear gradients (E14.S3).
//
// softmax's Jacobian dy_i/dx_j = y_i(delta_ij - y_j) is dense — 23 billion entries at V = 151,936 —
// so the backward never forms it. It needs only the product with G:
//
//     dL/dx_i = y_i . (G_i - sum_j G_j y_j)
//
// which is one weighted sum per lane. rms_normalize has a coupling term because normalisation ties
// a lane together; embedding's backward is a scatter-add, and nothing flows to the ids.
Variable silu(const Variable& x);
Variable gelu(const Variable& x);
Variable softmax(const Variable& x, size_t dim);
Variable rms_normalize(const Variable& x, float eps);
Variable embedding(const Variable& table, const std::vector<int64_t>& ids,
                   const core::Shape& id_shape);

// The rearrangements (E16.S2.T2). A transformer's forward pass moves data as well as computing on
// it — it splits heads, swaps axes and rotates channels — and the backward of a rearrangement is
// simply the inverse rearrangement. Nothing is differentiated here; the only available mistake is
// applying the inverse the wrong way round.
//
// transpose MATERIALISES its value rather than returning a view: several backwards on this tape
// index their input as a flat buffer, which is only correct for a contiguous tensor, and a silently
// wrong gradient is the worst failure this system has (risk R8).
//
// rope's backward is a rotation by the negative angle, because a rotation is orthogonal. The tables
// get no gradient: they are functions of position, not parameters.
Variable reshape(const Variable& x, const core::Shape& shape);
Variable transpose(const Variable& x, size_t a, size_t b);
Variable rope(const Variable& x, const core::Tensor& cosines, const core::Tensor& sines);

namespace detail
{
// Sums a gradient back to the shape of the input it belongs to.
core::Tensor unbroadcast(const core::Tensor& gradient, const core::Shape& target);
}

} // namespace veda::autograd

#endif //VEDA_VARIABLE_H
