#ifndef VEDA_ATTENTION_H
#define VEDA_ATTENTION_H

#include "Tensor.h"

#include <cstddef>

// The architecture layer: RoPE, attention, FFN, blocks, the transformer, the LM head.
//
// This is where head layout is a legitimate concern. nn::Linear produces out_features numbers and
// does not know they are heads (architecture §1: a module may not know why its caller needs it);
// deciding what those numbers mean happens here.
namespace veda::model
{

// [B, T, H*Dh] -> [B, H, T, Dh]
//
// A reinterpretation of memory, not a computation: reshape regroups the last axis, transpose swaps
// two axes by swapping their strides, and not one float moves. That is why sixteen heads cost the
// same as one.
//
// H comes before T in the result because attention works per head over the whole sequence: head 3
// needs its [T, Dh] block addressable as a unit. The result is not contiguous, which every op in
// veda::ops already handles.
//
// Throws std::invalid_argument if the input is not rank 3, or if its last dimension is not
// heads * head_dim.
core::Tensor split_heads(const core::Tensor& projected, size_t heads, size_t head_dim);

// [B, H, T, Dh] x [B, H, S, Dh] -> [B, H, T, S]
//
// Every query against every key: scores[b, h, t, s] is how much position t should care about
// position s, in head h. This is matmul_nt, not a transpose followed by matmul — Q and K share
// their last dimension, which is exactly the variant AD4 built. The transpose in "Q·Kt" is in the
// interpretation, not in the data.
//
// The keys carry their own length S: during prefill S == T, and with a KV cache (E12) one new
// query attends to all cached keys, so T is 1 while S grows.
//
// This is where the architecture's quadratic lives: the result holds B*H*T*S floats — 64 MB per
// layer at T = 1024, H = 16.
core::Tensor attention_scores(const core::Tensor& queries, const core::Tensor& keys);

// Divides the scores by sqrt(head_dim). Shape-preserving.
//
// Not a tidiness constant. With unit-variance components the dot product of two Dh-long vectors has
// variance Dh, so scores grow as sqrt(Dh) — around +-11 at Dh = 128 — while the relationships they
// measure have not changed. Softmax exponentiates differences, so scores that large collapse every
// row onto its single largest entry: the mechanism meant to blend the whole sequence becomes a hard
// lookup of one position.
//
// Dividing by sqrt(Dh) makes the score variance 1 at every head dimension. Dividing by Dh instead —
// the plausible-looking mistake — shrinks it to 1/sqrt(Dh), softmax flattens towards uniform, and
// attention blurs into an average. The wrong constant fails in the opposite direction and just as
// silently.
//
// It is head_dim, not hidden_size: the dot product runs over one head. Using D where Dh belongs is
// a silent factor of sqrt(H), and no shape check can see it.
//
// Applied before the mask (which adds -inf, and must not be scaled) and before softmax (which is
// what the scaling protects).
core::Tensor scale_scores(const core::Tensor& scores, size_t head_dim);

// A [queries, keys] matrix of 0 where attention is allowed and -inf where it is not.
//
// A decoder predicts the next token; if a position could see the one it is being trained to
// predict, the answer would be in the input and the model would learn to copy.
//
// Query t sits at absolute position keys - queries + t, so it may attend to key s when
// s <= keys - queries + t. During prefill queries == keys and that is the lower triangle; with a
// KV cache (E12) one new query attends to the whole history, and the row is all zeros.
//
// Throws if keys < queries: a query with no position to sit at is a caller bug.
core::Tensor causal_mask(size_t queries, size_t keys);

// scores + mask, over the last two dimensions of any rank, broadcast across batch and heads.
//
// -inf before softmax, never zeroing after (AD8): exp(-inf - m) is exactly 0, so a masked position
// contributes nothing AND is excluded before normalisation, leaving the remaining weights summing
// to 1. Zeroing afterwards would leave the row summing to less than 1 and silently scale the
// output down.
//
// Added rather than multiplied by a 0/1 matrix: multiplying would zero the scores, and a score of
// 0 is not "forbidden" but "neutral", which softmax gives a healthy probability to. Addition also
// composes — padding and window masks can be added to the same scores.
//
// The mask depends only on the two lengths, so one [T, S] triangle broadcasts over all B*H heads
// instead of being materialised B*H times.
core::Tensor apply_causal_mask(const core::Tensor& scores);

// [B, H, T, S] x [B, H, S, Dh] -> [B, H, T, Dh]
//
// Spends the weights: each output is a convex combination of the value vectors, so it always lies
// inside their hull — attention interpolates between what positions offer and can never extrapolate
// beyond them.
//
// A plain matmul, not matmul_nt: the shared dimension S is the last of the weights and the
// second-to-last of the values. The two variants sit one line apart in attention and mean different
// things — Q·Kt compares, probs·V combines.
core::Tensor attention_context(const core::Tensor& weights, const core::Tensor& values);

// [B, H, T, Dh] -> [B, T, H*Dh]
//
// Attention worked per head; the rest of the block works per token. The transpose back is free, the
// reshape after it is not: a token's heads are scattered T*Dh apart in memory, and a reshape can
// only relabel. So this is the one materialising copy in the attention path — 4 MB per layer at
// T = 1024, worth knowing about when E13 goes looking for time.
core::Tensor merge_heads(const core::Tensor& per_head);

// [B, Hkv, T, Dh] -> [B, H, T, Dh], each key/value head repeated H/Hkv times.
//
// Grouped-query attention: Qwen3-0.6B has 16 query heads and 8 key/value heads, so several queries
// ask slightly different questions of the same key/value pair. What that buys is the KV cache —
// 2*B*Hkv*T*Dh*4 bytes per layer, 7.5 GB rather than 15 GB at a 32k context across 28 layers.
// Query heads are cheap; cached keys are not, and long contexts are limited by exactly this.
//
// *** The grouping is contiguous blocks, not interleaving: kv_head(h) = h / (H / Hkv). ***
// With H = 4 and Hkv = 2, query heads 0 and 1 use KV head 0, and heads 2 and 3 use KV head 1. The
// interleaved alternative (h % Hkv) produces the same shapes and the same values in the wrong
// places, and nothing catches it but a test that names the expectation — so one does.
//
// This materialises: head h reads KV head h/rep, which is not a constant stride, so it cannot be a
// view. A zero-copy alternative exists — view Q as [B, Hkv, rep, T, Dh] and K as [B, Hkv, 1, T, Dh]
// and let matmul_nt's batch broadcasting repeat with a stride of zero — and it is exactly the kind
// of change E13 makes after measuring.
core::Tensor expand_kv_heads(const core::Tensor& kv, size_t heads);

} // namespace veda::model

#endif //VEDA_ATTENTION_H
