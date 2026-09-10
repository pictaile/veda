#include "AdamW.h"

#include "Shape.h"
#include "Storage.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace veda::optim
{

using core::Tensor;

AdamW::AdamW(std::vector<autograd::Variable> parameters, AdamWConfig config)
    : parameters_(std::move(parameters)), config_(config)
{
    if (config_.learning_rate < 0.0f || config_.eps <= 0.0f || config_.weight_decay < 0.0f)
    {
        throw std::invalid_argument("AdamW: learning rate, eps and weight decay must be positive");
    }
    if (config_.beta1 < 0.0f || config_.beta1 >= 1.0f || config_.beta2 < 0.0f ||
        config_.beta2 >= 1.0f)
    {
        throw std::invalid_argument("AdamW: betas must lie in [0, 1)");
    }

    // Allocated once, from the parameters' shapes — the state is as large as the model, twice.
    first_moment_.reserve(parameters_.size());
    second_moment_.reserve(parameters_.size());
    for (const autograd::Variable& parameter : parameters_)
    {
        first_moment_.emplace_back(parameter.value().shape());
        second_moment_.emplace_back(parameter.value().shape());
    }
}

void AdamW::step()
{
    ++steps_;

    // std::pow on the counter rather than a running product: bias correction is meant to be exact,
    // and a product accumulated in fp32 over thousands of steps drifts.
    const double bias1 = 1.0 - std::pow(static_cast<double>(config_.beta1),
                                        static_cast<double>(steps_));
    const double bias2 = 1.0 - std::pow(static_cast<double>(config_.beta2),
                                        static_cast<double>(steps_));

    for (size_t p = 0; p < parameters_.size(); ++p)
    {
        autograd::Variable& parameter = parameters_[p];
        if (!parameter.requires_grad())
        {
            continue;
        }

        float* value = parameter.node()->value.data();
        const float* gradient = parameter.node()->grad.data();
        float* m = first_moment_[p].data();
        float* v = second_moment_[p].data();

        for (size_t i = 0; i < first_moment_[p].numel(); ++i)
        {
            const float g = gradient[i];

            // The first moment: an average of the gradient. Consistent directions accumulate,
            // noise that changes sign between batches averages out.
            m[i] = config_.beta1 * m[i] + (1.0f - config_.beta1) * g;

            // The second: an average of the SQUARED gradient, so it is a scale, not a direction.
            v[i] = config_.beta2 * v[i] + (1.0f - config_.beta2) * g * g;

            // Both start at zero, so the early averages are pulled toward it: after one step m is
            // 0.1g and v is 0.001g^2, which estimate nothing. The correction is exact — at t = 1
            // it returns g and g^2 — and without it the first update is 3.16 times too large, at
            // precisely the moment the model is least able to survive it.
            const double m_hat = static_cast<double>(m[i]) / bias1;
            const double v_hat = static_cast<double>(v[i]) / bias2;

            // For a steady gradient this is g/|g| = +/-1, so the step is the learning rate whatever
            // the gradient's magnitude — the single fact that lets one learning rate serve a whole
            // network whose layers have wildly different gradient scales.
            const double adaptive = m_hat / (std::sqrt(v_hat) + config_.eps);

            // Decoupled: applied to the weight itself, never through v — and applied to the
            // value BEFORE the adaptive step, which is the reference ordering (AD7). The
            // difference is second order, and matching the reference matters more.
            value[i] -= config_.learning_rate * config_.weight_decay * value[i];

            value[i] -= config_.learning_rate * static_cast<float>(adaptive);
        }
    }
}

void AdamW::zero_grad()
{
    for (autograd::Variable& parameter : parameters_)
    {
        if (!parameter.requires_grad())
        {
            continue;
        }
        Tensor& gradient = parameter.node()->grad;
        for (size_t i = 0; i < gradient.numel(); ++i)
        {
            gradient.data()[i] = 0.0f;
        }
    }
}

} // namespace veda::optim
