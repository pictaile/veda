#ifndef VEDA_SCALED_DOT_PRODUCT_ATTENTION_H
#define VEDA_SCALED_DOT_PRODUCT_ATTENTION_H

#include "KVCache.h"
#include "Linear.h"
#include "RMSNorm.h"
#include "Tensor.h"

#include <cstddef>
#include <optional>

namespace veda::model
{

// One attention layer, in the order that matters:
//
//     project -> split heads -> QK-Norm -> RoPE -> scores -> scale -> mask -> softmax
//             -> blend with V -> merge heads -> output projection
//
// Assembling it here rather than leaving the calls to the transformer block makes that order a
// property of one place. Four of the steps can be swapped without any error appearing, and two of
// those swaps produce fluent, confident nonsense.
class ScaledDotProductAttention
{
public:
    // Views into the loader's buffer (AD2); nothing is allocated.
    struct Weights
    {
        nn::Linear q_proj;   // [H*Dh, D]
        nn::Linear k_proj;   // [Hkv*Dh, D]
        nn::Linear v_proj;   // [Hkv*Dh, D]
        nn::Linear o_proj;   // [D, H*Dh]

        // Qwen3 normalises each head's query and key over the Dh axis before the rotation, with
        // its own learned gamma [Dh]. Generic tutorials do not mention it and Llama does not do it;
        // the architecture (§4) names it as one of two Qwen3 specifics that silently produce wrong
        // numbers if missed. Optional, because models without it exist.
        std::optional<nn::RMSNorm> q_norm;
        std::optional<nn::RMSNorm> k_norm;
    };

    struct Settings
    {
        size_t heads = 1;
        size_t kv_heads = 1;
        size_t head_dim = 1;

        // Absent means no positional encoding at all — which is what E7's tests exercise, and what
        // makes this layer permutation-invariant.
        std::optional<float> rope_theta;
    };

    ScaledDotProductAttention(Weights weights, Settings settings);

    // [B, T, D] -> [B, T, D]. first_position is where this chunk sits in the sequence: 0 during
    // prefill, and the end of the history when generating with a KV cache (E12).
    core::Tensor forward(const core::Tensor& x, size_t first_position = 0) const;

    // The cached path (E12). The new positions' keys and values are appended to the cache, and the
    // scores are computed against the whole history: Q is [B,H,T,Dh] while K is [B,Hkv,used+T,Dh].
    //
    // Three earlier decisions make this need no new arithmetic: the causal mask takes T and S
    // separately (E8.S4.T5), attention_scores gives the keys their own length (E7.S2.T3), and RoPE
    // takes a first_position (E8.S2.T2). None of them changes here.
    core::Tensor forward_cached(const core::Tensor& x, KVCache& cache, size_t layer) const;

    size_t heads() const noexcept { return settings_.heads; }
    size_t kv_heads() const noexcept { return settings_.kv_heads; }
    size_t head_dim() const noexcept { return settings_.head_dim; }

private:
    Weights weights_;
    Settings settings_;
};

} // namespace veda::model

#endif //VEDA_SCALED_DOT_PRODUCT_ATTENTION_H
