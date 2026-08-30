#ifndef VEDA_LINEAR_H
#define VEDA_LINEAR_H

#include "Tensor.h"

#include <optional>

namespace veda::nn
{

// A projection: y = x * Wt (+ b).
//
// Called seven times per transformer block — q, k, v, o, gate, up, down — plus once for the LM
// head, and holding roughly 99% of the model's parameters.
//
// The weight is [out_features, in_features], the layout HuggingFace stores and AD4 keeps rather
// than transposing 2.4 GB at load. That is why the forward pass calls matmul_nt: the shared
// dimension is the last of both operands. Each row of the weight is a learned direction, and each
// output element says how much the input points along it.
//
// AD2: neither the weight nor the bias is allocated or copied here — both are views into the
// loader's buffer.
class Linear
{
public:
    explicit Linear(core::Tensor weight);
    Linear(core::Tensor weight, core::Tensor bias);

    // [..., in] -> [..., out]. Leading dimensions pass through untouched: matmul_nt iterates them,
    // so the same layer serves [T, D], [B, T, D] and [B, H, T, Dh] without knowing which it got.
    core::Tensor forward(const core::Tensor& x) const;

    size_t in_features() const { return weight_.shape()[1]; }
    size_t out_features() const { return weight_.shape()[0]; }
    bool has_bias() const noexcept { return bias_.has_value(); }

    const core::Tensor& weight() const noexcept { return weight_; }

private:
    core::Tensor weight_;
    // Optional rather than a tensor of zeros: Qwen3 sets attention_bias false and has none, and a
    // zero bias would cost a broadcast add on all 197 calls per token for nothing.
    std::optional<core::Tensor> bias_;
};

} // namespace veda::nn

#endif //VEDA_LINEAR_H
