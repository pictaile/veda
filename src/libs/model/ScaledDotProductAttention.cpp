#include "ScaledDotProductAttention.h"

#include "Attention.h"
#include "Rope.h"
#include "Shape.h"
#include "Profile.h"
#include "Softmax.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace veda::model
{

using core::Tensor;

ScaledDotProductAttention::ScaledDotProductAttention(Weights weights, Settings settings)
    : weights_(std::move(weights)), settings_(settings)
{
    const size_t heads = settings_.heads;
    const size_t kv_heads = settings_.kv_heads;
    const size_t head_dim = settings_.head_dim;

    if (heads == 0 || kv_heads == 0 || head_dim == 0)
    {
        throw std::invalid_argument("model::ScaledDotProductAttention: heads, kv_heads and head_dim "
                                    "must be positive");
    }

    // Grouped-query attention: several query heads share one key/value head. ModelConfig already
    // guarantees this divides for a real model (E4.S3.T6); a synthetic layer can still get it wrong.
    if (kv_heads > heads || heads % kv_heads != 0)
    {
        throw std::invalid_argument(
            "model::ScaledDotProductAttention: heads " + std::to_string(heads) +
            " do not divide into kv_heads " + std::to_string(kv_heads));
    }

    if (weights_.q_proj.out_features() != heads * head_dim)
    {
        throw std::invalid_argument("model::ScaledDotProductAttention: q_proj produces " +
                                    std::to_string(weights_.q_proj.out_features()) + ", expected " +
                                    std::to_string(heads * head_dim));
    }
    if (weights_.k_proj.out_features() != kv_heads * head_dim ||
        weights_.v_proj.out_features() != kv_heads * head_dim)
    {
        throw std::invalid_argument("model::ScaledDotProductAttention: k_proj and v_proj must "
                                    "produce " + std::to_string(kv_heads * head_dim));
    }
    if (weights_.o_proj.in_features() != heads * head_dim ||
        weights_.o_proj.out_features() != weights_.q_proj.in_features())
    {
        throw std::invalid_argument("model::ScaledDotProductAttention: o_proj must map " +
                                    std::to_string(heads * head_dim) + " back to " +
                                    std::to_string(weights_.q_proj.in_features()));
    }

    // The norms act per head, over the head dimension — not over the whole [H*Dh] row.
    if (weights_.q_norm && weights_.q_norm->hidden_size() != head_dim)
    {
        throw std::invalid_argument("model::ScaledDotProductAttention: q_norm gamma is [" +
                                    std::to_string(weights_.q_norm->hidden_size()) +
                                    "], expected [" + std::to_string(head_dim) + "]");
    }
    if (weights_.k_norm && weights_.k_norm->hidden_size() != head_dim)
    {
        throw std::invalid_argument("model::ScaledDotProductAttention: k_norm gamma is [" +
                                    std::to_string(weights_.k_norm->hidden_size()) +
                                    "], expected [" + std::to_string(head_dim) + "]");
    }
}

namespace
{
// Everything from the projections up to the point where K and V are ready. Shared by both paths so
// the cached one cannot drift from the reference.
struct Projected
{
    Tensor q;
    Tensor k;
    Tensor v;
};
} // namespace

Tensor ScaledDotProductAttention::forward(const Tensor& x, size_t first_position) const
{
    const profile::Scope scope("attention");
    const size_t head_dim = settings_.head_dim;

    Tensor q = split_heads(weights_.q_proj.forward(x), settings_.heads, head_dim);
    Tensor k = split_heads(weights_.k_proj.forward(x), settings_.kv_heads, head_dim);
    const Tensor v = split_heads(weights_.v_proj.forward(x), settings_.kv_heads, head_dim);

    // QK-Norm before the rotation, never after. RMSNorm divides by a magnitude and rotation
    // preserves magnitude, so the lengths agree either way — but gamma is per channel and rotation
    // moves information between channels within each pair, so the values do not. The wrong order
    // gives the right lengths, the right relative-distance property, and different numbers in every
    // channel: invisible without a reference (AD7).
    if (weights_.q_norm)
    {
        q = weights_.q_norm->forward(q);
    }
    if (weights_.k_norm)
    {
        k = weights_.k_norm->forward(k);
    }

    // V is neither normalised nor rotated: it carries what a position contributes, not how relevant
    // it is. Position belongs in Q and K.
    if (settings_.rope_theta)
    {
        const size_t positions = q.shape()[q.rank() - 2];
        const RopeTables tables =
            rope_tables(positions, head_dim, *settings_.rope_theta, first_position);
        q = apply_rope(q, tables);
        k = apply_rope(k, tables);
    }

    // Expanded after the rotation, not before: rotation depends only on position, so the two orders
    // agree — but this way only Hkv heads are rotated instead of H, at no cost in correctness.
    const Tensor expanded_k =
        settings_.kv_heads == settings_.heads ? k : expand_kv_heads(k, settings_.heads);
    const Tensor expanded_v =
        settings_.kv_heads == settings_.heads ? v : expand_kv_heads(v, settings_.heads);

    // Scale before mask (the mask adds -inf, which scaling must not touch), mask before softmax
    // (which turns -inf into an exact zero), softmax before the blend.
    const Tensor scores = attention_scores(q, expanded_k);
    const Tensor scaled = scale_scores(scores, head_dim);
    const Tensor masked = apply_causal_mask(scaled);
    const Tensor weights = ops::softmax(masked, masked.rank() - 1);

    return weights_.o_proj.forward(merge_heads(attention_context(weights, expanded_v)));
}

Tensor ScaledDotProductAttention::forward_cached(const Tensor& x, KVCache& cache, size_t layer) const
{
    const profile::Scope scope("attention");
    const size_t head_dim = settings_.head_dim;
    const size_t first_position = cache.used();

    Tensor q = split_heads(weights_.q_proj.forward(x), settings_.heads, head_dim);
    Tensor k = split_heads(weights_.k_proj.forward(x), settings_.kv_heads, head_dim);
    Tensor v = split_heads(weights_.v_proj.forward(x), settings_.kv_heads, head_dim);

    if (weights_.q_norm)
    {
        q = weights_.q_norm->forward(q);
    }
    if (weights_.k_norm)
    {
        k = weights_.k_norm->forward(k);
    }

    if (settings_.rope_theta)
    {
        const size_t positions = q.shape()[q.rank() - 2];
        const RopeTables tables =
            rope_tables(positions, head_dim, *settings_.rope_theta, first_position);
        q = apply_rope(q, tables);
        k = apply_rope(k, tables);
    }

    // Append first, then read the whole prefix back: the new position must attend to itself, which
    // is what the causal mask requires.
    cache.append(layer, k, v);

    // used() has not advanced yet — advancing is a separate call so that every layer appends at the
    // same offset — so the visible length is asked for explicitly. The new position must be
    // included: it has to attend to itself.
    const size_t new_positions = q.shape()[q.rank() - 2];
    const size_t visible = first_position + new_positions;
    const Tensor visible_k = cache.keys(layer, visible);
    const Tensor visible_v = cache.values(layer, visible);

    const Tensor expanded_k = settings_.kv_heads == settings_.heads
                                  ? visible_k
                                  : expand_kv_heads(visible_k, settings_.heads);
    const Tensor expanded_v = settings_.kv_heads == settings_.heads
                                  ? visible_v
                                  : expand_kv_heads(visible_v, settings_.heads);

    const Tensor scores = attention_scores(q, expanded_k);
    const Tensor scaled = scale_scores(scores, head_dim);
    const Tensor masked = apply_causal_mask(scaled);
    const Tensor weights = ops::softmax(masked, masked.rank() - 1);

    return weights_.o_proj.forward(merge_heads(attention_context(weights, expanded_v)));
}

} // namespace veda::model
