#include "Activations.h"

#include "Walk.h"

#include <cmath>
#include <numbers>

namespace veda::ops
{

using core::Tensor;

Tensor silu(const Tensor& tensor)
{
    return detail::map_elements(tensor, [](float x) { return x / (1.0f + std::exp(-x)); });
}

Tensor gelu(const Tensor& tensor)
{
    // sqrt(2/pi), the constant of the tanh approximation
    static constexpr float scale = 0.7978845608028654f;
    static constexpr float cubic = 0.044715f;

    return detail::map_elements(tensor, [](float x) {
        return 0.5f * x * (1.0f + std::tanh(scale * (x + cubic * x * x * x)));
    });
}

} // namespace veda::ops
