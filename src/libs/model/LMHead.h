#ifndef VEDA_LM_HEAD_H
#define VEDA_LM_HEAD_H

#include "Tensor.h"

#include <cstddef>

namespace veda::model
{

// One number per token in the vocabulary: how much the final hidden state points along that
// token's direction.
//
//     logit[v] = hidden . W[v]
//
// Unnormalised, unbounded, positive or negative, and meaningful only relative to the other logits
// of the same row. Softmax would turn them into a distribution — but that belongs to sampling
// (E11), and argmax does not need it: softmax is monotonic, so the largest logit is the largest
// probability. Normalising here would spend 151,936 exponentials per token to change nothing.
//
// *** Tied embeddings. *** Qwen3 sets tie_word_embeddings, so the head has no weights of its own:
// the embedding table [V, D] is reused as the output projection (§4 of the architecture). Beyond
// saving 155M parameters, it says that "what does this token mean" and "which token is this vector
// most like" are the same question asked in two directions. Some models do not accept that, and
// then lm_head.weight exists separately — which is why this takes a weight rather than a
// Transformer.
class LMHead
{
public:
    explicit LMHead(core::Tensor weight);   // [V, D]

    // [B, T, D] -> [B, T, V]: a prediction after every token, which is what training wants.
    core::Tensor forward(const core::Tensor& hidden) const;

    // [B, T, D] -> [B, V]: only the last position, which is what generating wants.
    //
    // Slices first and projects second. The head is about a third of a token's arithmetic — 155M
    // multiply-adds against the 353M of all 28 layers — so doing it for every position when only
    // the last is used costs a factor of T. That is not an optimisation; it is the difference
    // between generating and not.
    core::Tensor forward_last(const core::Tensor& hidden) const;

    size_t vocab_size() const { return weight_.shape()[0]; }
    size_t hidden_size() const { return weight_.shape()[1]; }
    const core::Tensor& weight() const noexcept { return weight_; }

private:
    core::Tensor weight_;
};

} // namespace veda::model

#endif //VEDA_LM_HEAD_H
