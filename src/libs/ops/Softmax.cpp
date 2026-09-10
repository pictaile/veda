#include "Softmax.h"

#include "Shape.h"
#include "Storage.h"
#include "Strides.h"
#include "Walk.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace veda::ops
{

using core::Shape;
using core::Tensor;

Tensor softmax(const Tensor& tensor, size_t dim)
{
    if (dim >= tensor.rank())
    {
        throw std::invalid_argument(
            "softmax: dimension " + std::to_string(dim) + " is out of range for shape " +
            tensor.shape().to_string());
    }

    const Shape& shape = tensor.shape();
    Tensor out{shape};
    if (shape.size() == 0)
    {
        return out;
    }

    const size_t length = shape[dim];                 // how long one lane is
    const size_t lanes = shape.size() / length;       // how many of them there are
    const size_t in_step = tensor.strides()[dim];
    const std::vector<size_t> out_strides = contiguous_strides(shape);
    const size_t out_step = out_strides[dim];

    const float* source = tensor.data();
    float* destination = out.data();

    std::vector<size_t> index(shape.rank(), 0);
    for (size_t lane = 0; lane < lanes; ++lane)
    {
        // where this lane starts in each tensor; the softmax axis is held at 0 and stepped below
        const size_t in_base = detail::offset_of(tensor.strides(), index);
        const size_t out_base = detail::offset_of(out_strides, index);

        // Three passes over the lane: maximum, sum, divide. Fusing them into an online normaliser
        // is possible and is exactly the kind of thing E13 measures before doing.
        float maximum = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < length; ++i)
        {
            const float value = source[in_base + i * in_step];
            maximum = value > maximum ? value : maximum;
        }

        // The sum accumulates in float, not double: Veda is fp32 throughout (AD1), and a wider
        // accumulator here would quietly diverge from the reference E3 compares against.
        float sum = 0.0f;
        for (size_t i = 0; i < length; ++i)
        {
            const float shifted = std::exp(source[in_base + i * in_step] - maximum);
            destination[out_base + i * out_step] = shifted;
            sum += shifted;
        }

        for (size_t i = 0; i < length; ++i)
        {
            destination[out_base + i * out_step] /= sum;
        }

        detail::advance_skipping(index, shape, dim);
    }

    return out;
}

Tensor log_softmax(const Tensor& tensor, size_t dim)
{
    if (dim >= tensor.rank())
    {
        throw std::invalid_argument(
            "log_softmax: dimension " + std::to_string(dim) + " is out of range for shape " +
            tensor.shape().to_string());
    }

    const Shape& shape = tensor.shape();
    Tensor out{shape};
    if (shape.size() == 0)
    {
        return out;
    }

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

        float maximum = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < length; ++i)
        {
            const float value = source[in_base + i * in_step];
            maximum = value > maximum ? value : maximum;
        }

        // A lane that is entirely -inf (every position masked) has no maximum to subtract from;
        // leaving it as -inf everywhere is the only honest answer, and inf - inf would give NaN.
        if (maximum == -std::numeric_limits<float>::infinity())
        {
            for (size_t i = 0; i < length; ++i)
            {
                destination[out_base + i * out_step] = -std::numeric_limits<float>::infinity();
            }
            detail::advance_skipping(index, shape, dim);
            continue;
        }

        float sum = 0.0f;
        for (size_t i = 0; i < length; ++i)
        {
            sum += std::exp(source[in_base + i * in_step] - maximum);
        }
        const float log_sum = std::log(sum);

        // The subtraction, not a division: x - m - log_sum stays finite however small the
        // probability is, which is the whole point of the op.
        for (size_t i = 0; i < length; ++i)
        {
            destination[out_base + i * out_step] =
                source[in_base + i * in_step] - maximum - log_sum;
        }

        detail::advance_skipping(index, shape, dim);
    }

    return out;
}

} // namespace veda::ops
