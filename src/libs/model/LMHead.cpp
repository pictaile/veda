#include "LMHead.h"

#include "Matmul.h"
#include "Profile.h"
#include "Shape.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace veda::model
{

using core::Shape;
using core::Tensor;

LMHead::LMHead(Tensor weight) : weight_(std::move(weight))
{
    if (weight_.rank() != 2)
    {
        throw std::invalid_argument("model::LMHead: the weight must be rank 2 [V, D], got " +
                                    weight_.shape().to_string());
    }
}

Tensor LMHead::forward(const Tensor& hidden) const
{
    if (hidden.rank() < 2 || hidden.shape()[hidden.rank() - 1] != hidden_size())
    {
        throw std::invalid_argument("model::LMHead: hidden state " + hidden.shape().to_string() +
                                    " does not end in " + std::to_string(hidden_size()));
    }

    // The weight is [V, D] and the hidden state ends in D — the shared dimension is the last of
    // both, which is exactly matmul_nt (AD4). No transpose is constructed.
    return ops::matmul_nt(hidden, weight_);
}

Tensor LMHead::forward_last(const Tensor& hidden) const
{
    const profile::Scope scope("lm_head");
    if (hidden.rank() != 3 || hidden.shape()[2] != hidden_size())
    {
        throw std::invalid_argument("model::LMHead: expected [B, T, D] ending in " +
                                    std::to_string(hidden_size()) + ", got " +
                                    hidden.shape().to_string());
    }

    const size_t batch = hidden.shape()[0];
    const size_t positions = hidden.shape()[1];
    if (positions == 0)
    {
        throw std::invalid_argument("model::LMHead: there is no last position in an empty sequence");
    }

    // Slice, then project — never the other way round.
    const Tensor last = hidden.slice(1, positions - 1, 1);   // [B, 1, D]
    return ops::matmul_nt(last, weight_).reshape(Shape({batch, vocab_size()}));
}

} // namespace veda::model
