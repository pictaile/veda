#include "Transformer.h"

#include <utility>

namespace veda::model
{

using core::Shape;
using core::Tensor;

Transformer::Transformer(Weights weights) : weights_(std::move(weights)) {}

Tensor Transformer::forward(const std::vector<int64_t>& ids, const Shape& id_shape,
                            size_t first_position) const
{
    Tensor hidden = weights_.embedding.forward(ids, id_shape);

    // No KV cache yet (E12): every call reprocesses the whole prefix.
    for (const TransformerBlock& block : weights_.blocks)
    {
        hidden = block.forward(hidden, first_position);
    }

    return weights_.final_norm.forward(hidden);
}

Tensor Transformer::forward_cached(const std::vector<int64_t>& ids, const Shape& id_shape,
                                   KVCache& cache) const
{
    Tensor hidden = weights_.embedding.forward(ids, id_shape);

    const size_t new_positions = id_shape.rank() >= 2 ? id_shape[id_shape.rank() - 1] : ids.size();
    for (size_t layer = 0; layer < weights_.blocks.size(); ++layer)
    {
        hidden = weights_.blocks[layer].forward_cached(hidden, cache, layer);
    }

    // Advanced once, after every layer has appended at the same offset.
    cache.advance(new_positions);

    return weights_.final_norm.forward(hidden);
}

Transformer::Trace Transformer::forward_capturing(const std::vector<int64_t>& ids,
                                                  const Shape& id_shape,
                                                  size_t first_position) const
{
    Tensor embeddings = weights_.embedding.forward(ids, id_shape);

    std::vector<Tensor> layer_outputs;
    layer_outputs.reserve(weights_.blocks.size());

    Tensor hidden = embeddings;
    for (const TransformerBlock& block : weights_.blocks)
    {
        hidden = block.forward(hidden, first_position);
        layer_outputs.push_back(hidden);
    }

    Tensor final_hidden = weights_.final_norm.forward(hidden);

    return Trace{std::move(embeddings), std::move(layer_outputs), std::move(final_hidden)};
}

} // namespace veda::model
