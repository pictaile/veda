#include "Compare.h"

#include "Shape.h"
#include "Storage.h"
#include "Walk.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace veda::testing
{

using core::Shape;
using core::Tensor;

namespace
{

std::string render_index(const std::vector<size_t>& index)
{
    std::string out = "(";
    for (size_t k = 0; k < index.size(); ++k)
    {
        out += (k > 0 ? ", " : "") + std::to_string(index[k]);
    }
    return out + ")";
}

std::string render(float value)
{
    if (std::isnan(value))
    {
        return "nan";
    }
    if (std::isinf(value))
    {
        return value > 0.0f ? "inf" : "-inf";
    }
    return std::to_string(value);
}

// How far past the tolerance this pair is. Negative means it passes; the largest value is the
// worst offender. Non-finite pairs are decided before this is called.
float excess(float actual, float expected, float rtol, float atol)
{
    return std::fabs(actual - expected) - (atol + rtol * std::fabs(expected));
}

} // namespace

Difference compare_close(const Tensor& actual, const Tensor& expected, float rtol, float atol)
{
    Difference difference;
    difference.total = actual.numel();

    if (actual.shape() != expected.shape())
    {
        difference.equal = false;
        difference.message = "tensors differ in shape: actual " + actual.shape().to_string() +
                             ", expected " + expected.shape().to_string();
        return difference;
    }

    const Shape& shape = actual.shape();
    if (shape.size() == 0)
    {
        return difference;   // two empty tensors of the same shape agree
    }

    std::vector<size_t> index(shape.rank(), 0);
    std::vector<size_t> worst_index;
    float worst_excess = 0.0f;
    float worst_actual = 0.0f;
    float worst_expected = 0.0f;

    for (size_t i = 0; i < shape.size(); ++i)
    {
        const float left = ops::detail::read(actual, index);
        const float right = ops::detail::read(expected, index);

        // Non-finite values first: the tolerance rule is wrong about both cases.
        bool failed = false;
        float distance = 0.0f;
        if (std::isnan(left) || std::isnan(right))
        {
            failed = true;
            distance = std::numeric_limits<float>::infinity();
        }
        else if (std::isinf(left) || std::isinf(right))
        {
            failed = left != right;
            distance = failed ? std::numeric_limits<float>::infinity() : 0.0f;
        }
        else
        {
            distance = excess(left, right, rtol, atol);
            failed = distance > 0.0f;
        }

        if (failed)
        {
            ++difference.mismatches;
            if (worst_index.empty() || distance > worst_excess)
            {
                worst_index = index;
                worst_excess = distance;
                worst_actual = left;
                worst_expected = right;
            }
        }

        ops::detail::advance(index, shape);
    }

    if (difference.mismatches == 0)
    {
        return difference;
    }

    difference.equal = false;
    difference.index = worst_index;
    difference.actual = worst_actual;
    difference.expected = worst_expected;
    difference.absolute = std::fabs(worst_actual - worst_expected);
    difference.relative = worst_expected != 0.0f ? difference.absolute / std::fabs(worst_expected)
                                                 : std::numeric_limits<float>::infinity();

    difference.message =
        "tensors differ at index " + render_index(worst_index) + " of " + shape.to_string() +
        ": actual " + render(worst_actual) + ", expected " + render(worst_expected) +
        "\n  absolute " + render(difference.absolute) + ", relative " + render(difference.relative) +
        ", tolerance rtol " + render(rtol) + " atol " + render(atol) + "\n  " +
        std::to_string(difference.mismatches) + " of " + std::to_string(difference.total) +
        " elements differ";

    return difference;
}

void assert_close(const Tensor& actual, const Tensor& expected, float rtol, float atol)
{
    const Difference difference = compare_close(actual, expected, rtol, atol);
    if (!difference.equal)
    {
        throw std::runtime_error(difference.message);
    }
}

} // namespace veda::testing
