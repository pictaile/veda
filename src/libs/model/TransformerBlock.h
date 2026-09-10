#ifndef VEDA_TRANSFORMER_BLOCK_H
#define VEDA_TRANSFORMER_BLOCK_H

#include "FeedForward.h"
#include "KVCache.h"
#include "RMSNorm.h"
#include "ScaledDotProductAttention.h"
#include "Tensor.h"

#include <cstddef>

namespace veda::model
{

// The unit the model is 28 copies of.
//
//     h   = x + attention(input_norm(x))
//     out = h + feed_forward(post_attention_norm(h))
//
// Two sub-layers, each wrapped the same way: normalise, transform, add the input back. The addition
// is what makes depth work — a sub-layer outputs a *change* rather than a replacement, and there is
// a path from layer 0's input to layer 27's output that passes through no matrix at all.
//
// Pre-norm, not post-norm: the normalisation happens inside the wrapper, so the skip path stays
// untouched all the way down. The original transformer normalised after the addition, which puts a
// norm on the residual path at every layer and makes deep stacks finicky to train.
//
// Consequence for inference: the residual stream grows in magnitude as layers add to it. That is
// expected — the next block normalises before reading it, and the final norm cleans it up before
// the LM head.
//
// Qwen3's names are input_layernorm (before attention) and post_attention_layernorm (before the
// FFN, despite the name). Both are RMSNorm, both are [D], and swapping them is a live possibility
// that only a reference comparison catches.
class TransformerBlock
{
public:
    struct Weights
    {
        nn::RMSNorm input_norm;
        ScaledDotProductAttention attention;
        nn::RMSNorm post_attention_norm;
        FeedForward feed_forward;
    };

    explicit TransformerBlock(Weights weights);

    // [B, T, D] -> [B, T, D]. first_position is where this chunk sits in the sequence.
    core::Tensor forward(const core::Tensor& x, size_t first_position = 0) const;

    // The same two sub-layers, with attention reading and writing the cache (E12).
    core::Tensor forward_cached(const core::Tensor& x, KVCache& cache, size_t layer) const;

    size_t hidden_size() const { return weights_.input_norm.hidden_size(); }

private:
    Weights weights_;
};

} // namespace veda::model

#endif //VEDA_TRANSFORMER_BLOCK_H
