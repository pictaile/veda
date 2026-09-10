#include "TransformerBlock.h"

#include "Elementwise.h"
#include "Shape.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace veda::model
{

using core::Tensor;

TransformerBlock::TransformerBlock(Weights weights) : weights_(std::move(weights))
{
    if (weights_.input_norm.hidden_size() != weights_.post_attention_norm.hidden_size())
    {
        throw std::invalid_argument("model::TransformerBlock: the two norms disagree on width: " +
                                    std::to_string(weights_.input_norm.hidden_size()) + " and " +
                                    std::to_string(weights_.post_attention_norm.hidden_size()));
    }
    if (weights_.feed_forward.hidden_size() != weights_.input_norm.hidden_size())
    {
        throw std::invalid_argument("model::TransformerBlock: the feed-forward takes " +
                                    std::to_string(weights_.feed_forward.hidden_size()) +
                                    ", the norms are " +
                                    std::to_string(weights_.input_norm.hidden_size()));
    }
}

Tensor TransformerBlock::forward(const Tensor& x, size_t first_position) const
{
    // Normalise, transform, add — twice. The addition is last, so the skip path is untouched.
    const Tensor attended =
        weights_.attention.forward(weights_.input_norm.forward(x), first_position);
    const Tensor h = ops::add(x, attended);

    const Tensor transformed = weights_.feed_forward.forward(weights_.post_attention_norm.forward(h));
    return ops::add(h, transformed);
}

Tensor TransformerBlock::forward_cached(const Tensor& x, KVCache& cache, size_t layer) const
{
    const Tensor attended = weights_.attention.forward_cached(weights_.input_norm.forward(x), cache,
                                                              layer);
    const Tensor h = ops::add(x, attended);

    const Tensor transformed = weights_.feed_forward.forward(weights_.post_attention_norm.forward(h));
    return ops::add(h, transformed);
}

} // namespace veda::model
