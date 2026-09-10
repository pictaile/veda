#include "GradientCheck.h"

#include "Tensor.h"

#include <cmath>
#include <string>

namespace veda::autograd
{

using core::Tensor;

GradientCheck check_gradient(const std::function<Variable(const Variable&)>& loss_of,
                             const Tensor& at, float step, double tolerance)
{
    GradientCheck result;

    // The analytic gradient, once.
    Variable x = Variable::leaf(core::contiguous(at));
    Variable loss = loss_of(x);
    if (loss.value().numel() != 1)
    {
        result.ok = false;
        result.message = "check_gradient: the function must return a scalar, got " +
                         loss.value().shape().to_string();
        return result;
    }
    loss.backward();

    for (size_t i = 0; i < at.numel(); ++i)
    {
        Tensor perturbed = core::contiguous(at);

        perturbed.data()[i] += step;
        const Variable up = loss_of(Variable::leaf(core::contiguous(perturbed), false));

        perturbed.data()[i] -= 2.0f * step;
        const Variable down = loss_of(Variable::leaf(core::contiguous(perturbed), false));

        const double numeric = (static_cast<double>(up.value().data()[0]) - down.value().data()[0]) /
                               (2.0 * static_cast<double>(step));
        const double analytic = x.grad().data()[i];

        const double scale = std::max({std::fabs(numeric), std::fabs(analytic), 1e-3});
        const double relative = std::fabs(numeric - analytic) / scale;

        if (relative > result.worst_relative)
        {
            result.worst_relative = relative;
            result.worst_index = i;
            result.analytic = analytic;
            result.numeric = numeric;
        }
    }

    result.ok = result.worst_relative <= tolerance;
    if (!result.ok)
    {
        result.message = "gradient mismatch at index " + std::to_string(result.worst_index) +
                         ": analytic " + std::to_string(result.analytic) + ", numeric " +
                         std::to_string(result.numeric) + ", relative " +
                         std::to_string(result.worst_relative);
    }
    return result;
}

} // namespace veda::autograd
