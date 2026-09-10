#ifndef VEDA_FEED_FORWARD_H
#define VEDA_FEED_FORWARD_H

#include "Linear.h"
#include "Tensor.h"

#include <cstddef>

namespace veda::model
{

// The other half of a transformer block.
//
//     attention   mixes information BETWEEN positions
//     FeedForward transforms each position INDEPENDENTLY — no token sees any other
//
// Attention decides what is relevant; this decides what to do with it. One token's output depends
// on that token's vector and nothing else, which is why this layer needs no mask, no positions and
// no cache.
//
// SwiGLU: three matrices rather than two.
//
//     gate = silu(W_gate . x)      [D] -> [F]
//     up   =      W_up   . x       [D] -> [F]
//     out  = W_down . (gate * up)  [F] -> [D]
//
// The activation is applied to one branch only, and that branch multiplies the other: for each of
// the F channels, one branch decides how much of the other passes through — a learned, per-channel,
// input-dependent valve. The wide middle (F = 3D in Qwen3) is where combinations that are not
// linearly separable in D dimensions have room to exist.
//
// 264M of Qwen3-0.6B's ~596M parameters are here, across 28 layers.
class FeedForward
{
public:
    struct Weights
    {
        nn::Linear gate_proj;   // [F, D]
        nn::Linear up_proj;     // [F, D]
        nn::Linear down_proj;   // [D, F]
    };

    explicit FeedForward(Weights weights);

    // [..., D] -> [..., D]. Leading dimensions pass through untouched.
    core::Tensor forward(const core::Tensor& x) const;

    size_t hidden_size() const { return weights_.gate_proj.in_features(); }
    size_t inner_size() const { return weights_.gate_proj.out_features(); }

private:
    Weights weights_;
};

} // namespace veda::model

#endif //VEDA_FEED_FORWARD_H
