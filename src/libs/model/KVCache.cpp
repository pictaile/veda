#include "KVCache.h"

#include "Shape.h"
#include "Storage.h"

#include <stdexcept>
#include <string>

namespace veda::model
{

using core::Shape;
using core::Tensor;

namespace
{

void copy_into(Tensor& destination, const Tensor& source, size_t offset)
{
    const size_t batch = source.shape()[0];
    const size_t heads = source.shape()[1];
    const size_t positions = source.shape()[2];
    const size_t head_dim = source.shape()[3];

    for (size_t b = 0; b < batch; ++b)
    {
        for (size_t h = 0; h < heads; ++h)
        {
            for (size_t t = 0; t < positions; ++t)
            {
                for (size_t d = 0; d < head_dim; ++d)
                {
                    const size_t target = destination.offset() + b * destination.strides()[0] +
                                          h * destination.strides()[1] +
                                          (offset + t) * destination.strides()[2] +
                                          d * destination.strides()[3];
                    destination.storage()->data()[target] = source.at({b, h, t, d});
                }
            }
        }
    }
}

} // namespace

KVCache::KVCache(size_t layers, size_t batch, size_t kv_heads, size_t max_positions,
                 size_t head_dim)
    : max_positions_(max_positions)
{
    if (layers == 0 || batch == 0 || kv_heads == 0 || max_positions == 0 || head_dim == 0)
    {
        throw std::invalid_argument("model::KVCache: every dimension must be positive");
    }

    keys_.reserve(layers);
    values_.reserve(layers);
    for (size_t layer = 0; layer < layers; ++layer)
    {
        keys_.emplace_back(Shape({batch, kv_heads, max_positions, head_dim}));
        values_.emplace_back(Shape({batch, kv_heads, max_positions, head_dim}));
    }
}

void KVCache::append(size_t layer, const Tensor& keys, const Tensor& values)
{
    if (layer >= keys_.size())
    {
        throw std::out_of_range("model::KVCache: no layer " + std::to_string(layer));
    }
    if (keys.rank() != 4 || values.rank() != 4 || keys.shape() != values.shape())
    {
        throw std::invalid_argument("model::KVCache: keys and values must be [B, Hkv, T, Dh] and "
                                    "agree, got " + keys.shape().to_string() + " and " +
                                    values.shape().to_string());
    }

    const Tensor& storage = keys_[layer];
    for (size_t axis : {size_t{0}, size_t{1}, size_t{3}})
    {
        if (keys.shape()[axis] != storage.shape()[axis])
        {
            throw std::invalid_argument("model::KVCache: " + keys.shape().to_string() +
                                        " does not fit the cache's " + storage.shape().to_string());
        }
    }

    const size_t positions = keys.shape()[2];
    if (used_ + positions > max_positions_)
    {
        throw std::runtime_error("model::KVCache: appending " + std::to_string(positions) +
                                 " positions to " + std::to_string(used_) +
                                 " would exceed the limit of " + std::to_string(max_positions_));
    }

    copy_into(keys_[layer], keys, used_);
    copy_into(values_[layer], values, used_);
}

Tensor KVCache::keys(size_t layer) const
{
    return keys(layer, used_);
}

Tensor KVCache::values(size_t layer) const
{
    return values(layer, used_);
}

Tensor KVCache::keys(size_t layer, size_t length) const
{
    if (layer >= keys_.size())
    {
        throw std::out_of_range("model::KVCache: no layer " + std::to_string(layer));
    }
    if (length > max_positions_)
    {
        throw std::out_of_range("model::KVCache: asked for " + std::to_string(length) +
                                " positions, the cache holds " + std::to_string(max_positions_));
    }
    return keys_[layer].slice(2, 0, length);
}

Tensor KVCache::values(size_t layer, size_t length) const
{
    if (layer >= values_.size())
    {
        throw std::out_of_range("model::KVCache: no layer " + std::to_string(layer));
    }
    if (length > max_positions_)
    {
        throw std::out_of_range("model::KVCache: asked for " + std::to_string(length) +
                                " positions, the cache holds " + std::to_string(max_positions_));
    }
    return values_[layer].slice(2, 0, length);
}

void KVCache::advance(size_t positions)
{
    if (used_ + positions > max_positions_)
    {
        throw std::runtime_error("model::KVCache: advancing past the limit");
    }
    used_ += positions;
}

} // namespace veda::model
