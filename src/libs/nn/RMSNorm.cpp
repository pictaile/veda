#include "RMSNorm.h"

#include "Elementwise.h"
#include "Normalize.h"
#include "Shape.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace veda::nn
{

using core::Tensor;

RMSNorm::RMSNorm(Tensor gamma, float eps) : gamma_(std::move(gamma)), eps_(eps)
{
    if (gamma_.rank() != 1)
    {
        throw std::invalid_argument("nn::RMSNorm: gamma must be rank 1 [D], got " +
                                    gamma_.shape().to_string());
    }
    if (!(eps_ > 0.0f))
    {
        throw std::invalid_argument("nn::RMSNorm: eps is " + std::to_string(eps_) +
                                    ", must be positive");
    }
}

Tensor RMSNorm::forward(const Tensor& x) const
{
    if (x.rank() == 0 || x.shape()[x.rank() - 1] != hidden_size())
    {
        throw std::invalid_argument("nn::RMSNorm: input " + x.shape().to_string() +
                                    " does not end in the gamma length " +
                                    std::to_string(hidden_size()));
    }

    // The maths lives in ops; this layer only scales. There is no square root here on purpose —
    // if there were, the split the architecture asked for would have leaked.
    return ops::mul(ops::rms_normalize(x, eps_), gamma_);
}

} // namespace veda::nn
