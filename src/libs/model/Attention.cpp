#include "Attention.h"

#include "Elementwise.h"
#include "Matmul.h"
#include "Shape.h"
#include "Storage.h"

#include <cmath>
#include <limits>

#include <stdexcept>
#include <string>
#include <vector>

namespace veda::model
{

using core::Shape;
using core::Tensor;

Tensor split_heads(const Tensor& projected, size_t heads, size_t head_dim)
{
    if (projected.rank() != 3)
    {
        throw std::invalid_argument("model::split_heads: expected [B, T, H*Dh], got " +
                                    projected.shape().to_string());
    }
    if (heads == 0 || head_dim == 0)
    {
        throw std::invalid_argument("model::split_heads: heads and head_dim must be positive, got " +
                                    std::to_string(heads) + " and " + std::to_string(head_dim));
    }

    const size_t batch = projected.shape()[0];
    const size_t positions = projected.shape()[1];
    const size_t channels = projected.shape()[2];

    if (channels != heads * head_dim)
    {
        throw std::invalid_argument(
            "model::split_heads: last dimension " + std::to_string(channels) + " is not " +
            std::to_string(heads) + " heads of " + std::to_string(head_dim));
    }

    // [B, T, H*Dh] -> [B, T, H, Dh] -> [B, H, T, Dh]. Both steps are views (E1.S2.T5, T6).
    return projected.reshape(Shape({batch, positions, heads, head_dim})).transpose(1, 2);
}

Tensor attention_scores(const Tensor& queries, const Tensor& keys)
{
    if (queries.rank() < 3 || keys.rank() < 3)
    {
        throw std::invalid_argument(
            "model::attention_scores: expected [.., H, T, Dh], got " + queries.shape().to_string() +
            " and " + keys.shape().to_string());
    }
    if (queries.shape()[queries.rank() - 1] != keys.shape()[keys.rank() - 1])
    {
        throw std::invalid_argument("model::attention_scores: head dimensions disagree: " +
                                    queries.shape().to_string() + " and " +
                                    keys.shape().to_string());
    }

    // The heads ride along as batch dimensions: matmul_nt iterates everything left of the last two
    // axes, so B*H independent [T, Dh] x [Dh, S] products happen with no loop here.
    return ops::matmul_nt(queries, keys);
}

Tensor scale_scores(const Tensor& scores, size_t head_dim)
{
    if (head_dim == 0)
    {
        throw std::invalid_argument("model::scale_scores: head_dim must be positive");
    }

    // One division, then B*H*T*S multiplications by the reciprocal.
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    return ops::mul(scores, scale);
}

core::Tensor causal_mask(size_t queries, size_t keys)
{
    if (keys < queries)
    {
        throw std::invalid_argument("model::causal_mask: " + std::to_string(queries) +
                                    " queries against only " + std::to_string(keys) + " keys");
    }

    Tensor mask{Shape({queries, keys})};
    const float forbidden = -std::numeric_limits<float>::infinity();

    // Query t is at absolute position (keys - queries + t): during prefill that is t, and with a
    // cache the single new query sits at the end of the history.
    const size_t offset = keys - queries;
    float* destination = mask.data();
    for (size_t t = 0; t < queries; ++t)
    {
        const size_t last_allowed = offset + t;
        for (size_t s = 0; s < keys; ++s)
        {
            destination[t * keys + s] = s <= last_allowed ? 0.0f : forbidden;
        }
    }
    return mask;
}

Tensor apply_causal_mask(const Tensor& scores)
{
    if (scores.rank() < 2)
    {
        throw std::invalid_argument("model::apply_causal_mask: expected [.., T, S], got " +
                                    scores.shape().to_string());
    }

    const size_t queries = scores.shape()[scores.rank() - 2];
    const size_t keys = scores.shape()[scores.rank() - 1];

    // One triangle, broadcast over batch and heads — not B*H copies of it.
    return ops::add(scores, causal_mask(queries, keys));
}

Tensor attention_context(const Tensor& weights, const Tensor& values)
{
    if (weights.rank() < 3 || values.rank() < 3)
    {
        throw std::invalid_argument("model::attention_context: expected [.., T, S] and [.., S, Dh], got " +
                                    weights.shape().to_string() + " and " + values.shape().to_string());
    }
    if (weights.shape()[weights.rank() - 1] != values.shape()[values.rank() - 2])
    {
        throw std::invalid_argument("model::attention_context: key counts disagree: " +
                                    weights.shape().to_string() + " and " +
                                    values.shape().to_string());
    }

    return ops::matmul(weights, values);
}

Tensor merge_heads(const Tensor& per_head)
{
    if (per_head.rank() != 4)
    {
        throw std::invalid_argument("model::merge_heads: expected [B, H, T, Dh], got " +
                                    per_head.shape().to_string());
    }

    const size_t batch = per_head.shape()[0];
    const size_t heads = per_head.shape()[1];
    const size_t positions = per_head.shape()[2];
    const size_t head_dim = per_head.shape()[3];

    // transpose is free; the reshape after it needs dense memory, so the copy happens here.
    const Tensor by_position = core::contiguous(per_head.transpose(1, 2));
    return by_position.reshape(Shape({batch, positions, heads * head_dim}));
}

Tensor expand_kv_heads(const Tensor& kv, size_t heads)
{
    if (kv.rank() != 4)
    {
        throw std::invalid_argument("model::expand_kv_heads: expected [B, Hkv, T, Dh], got " +
                                    kv.shape().to_string());
    }

    const size_t batch = kv.shape()[0];
    const size_t kv_heads = kv.shape()[1];
    const size_t positions = kv.shape()[2];
    const size_t head_dim = kv.shape()[3];

    if (heads == 0 || kv_heads == 0 || heads % kv_heads != 0)
    {
        throw std::invalid_argument("model::expand_kv_heads: " + std::to_string(heads) +
                                    " query heads do not divide into " + std::to_string(kv_heads) +
                                    " kv heads");
    }

    const size_t repeats = heads / kv_heads;

    Tensor out{Shape({batch, heads, positions, head_dim})};
    if (out.numel() == 0)
    {
        return out;
    }

    float* destination = out.data();
    for (size_t b = 0; b < batch; ++b)
    {
        for (size_t h = 0; h < heads; ++h)
        {
            // Contiguous blocks: query heads h and h+1 of a group share one KV head.
            const size_t source_head = h / repeats;

            for (size_t t = 0; t < positions; ++t)
            {
                const size_t source_base = kv.offset() + b * kv.strides()[0] +
                                           source_head * kv.strides()[1] + t * kv.strides()[2];
                const size_t out_base = ((b * heads + h) * positions + t) * head_dim;

                for (size_t d = 0; d < head_dim; ++d)
                {
                    destination[out_base + d] =
                        kv.storage()->data()[source_base + d * kv.strides()[3]];
                }
            }
        }
    }

    return out;
}

} // namespace veda::model
