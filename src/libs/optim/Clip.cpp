#include "Clip.h"

#include "Storage.h"
#include "Tensor.h"

#include <cmath>
#include <stdexcept>

namespace veda::optim
{

float clip_gradient_norm(const std::vector<autograd::Variable>& parameters, float max_norm)
{
    if (!(max_norm > 0.0f))
    {
        throw std::invalid_argument("clip_gradient_norm: max_norm must be positive");
    }

    // Accumulated in double: the values are squared before they are summed, and a real model has
    // hundreds of millions of them.
    double squares = 0.0;
    for (const autograd::Variable& parameter : parameters)
    {
        if (!parameter.requires_grad())
        {
            continue;
        }
        const core::Tensor& gradient = parameter.grad();
        for (size_t i = 0; i < gradient.numel(); ++i)
        {
            const double g = gradient.data()[i];
            squares += g * g;
        }
    }

    const double total = std::sqrt(squares);
    if (!std::isfinite(total))
    {
        return static_cast<float>(total);   // reported, not spread
    }
    if (total <= static_cast<double>(max_norm))
    {
        return static_cast<float>(total);   // nothing is written at all
    }

    const float scale = static_cast<float>(static_cast<double>(max_norm) / total);
    for (const autograd::Variable& parameter : parameters)
    {
        if (!parameter.requires_grad())
        {
            continue;
        }
        core::Tensor& gradient = parameter.node()->grad;
        for (size_t i = 0; i < gradient.numel(); ++i)
        {
            gradient.data()[i] *= scale;
        }
    }

    return static_cast<float>(total);
}

} // namespace veda::optim
