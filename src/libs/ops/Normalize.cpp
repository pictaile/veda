#include "Normalize.h"

#include "Shape.h"
#include "Storage.h"
#include "Strides.h"
#include "Walk.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace veda::ops
{

using core::Shape;
using core::Tensor;

Tensor rms_normalize(const Tensor& tensor, float eps)
{
    if (tensor.rank() == 0)
    {
        throw std::invalid_argument("rms_normalize: a rank-0 tensor has no last dimension");
    }

    const Shape& shape = tensor.shape();
    Tensor out{shape};
    if (shape.size() == 0)
    {
        return out;
    }

    const size_t dim = shape.rank() - 1;
    const size_t length = shape[dim];
    const size_t lanes = shape.size() / length;
    const size_t in_step = tensor.strides()[dim];
    const std::vector<size_t> out_strides = contiguous_strides(shape);
    const size_t out_step = out_strides[dim];

    const float* source = tensor.data();
    float* destination = out.data();

    std::vector<size_t> index(shape.rank(), 0);
    for (size_t lane = 0; lane < lanes; ++lane)
    {
        const size_t in_base = detail::offset_of(tensor.strides(), index);
        const size_t out_base = detail::offset_of(out_strides, index);

        // Accumulated in float, not double: Veda is fp32 throughout (AD1), and a wider accumulator
        // here would quietly diverge from the reference E3 compares against.
        float sum_of_squares = 0.0f;
        for (size_t i = 0; i < length; ++i)
        {
            const float value = source[in_base + i * in_step];
            sum_of_squares += value * value;
        }

        // eps sits inside the root, added to the mean square: it stops a division by zero for an
        // all-zero vector and bounds the amplification of a nearly-zero one.
        const float rms = std::sqrt(sum_of_squares / static_cast<float>(length) + eps);

        for (size_t i = 0; i < length; ++i)
        {
            destination[out_base + i * out_step] = source[in_base + i * in_step] / rms;
        }

        detail::advance_skipping(index, shape, dim);
    }

    return out;
}

} // namespace veda::ops
