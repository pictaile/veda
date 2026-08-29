#include "Strides.h"

#include <cassert>

namespace veda::core
{

std::vector<size_t> contiguous_strides(const Shape& shape)
{
    const size_t rank = shape.rank();
    std::vector<size_t> strides(rank);

    // Back to front: the last dimension is packed one element apart, and each dimension to the
    // left steps over a whole block of the dimensions to its right.
    size_t stride = 1;
    for (size_t k = rank; k-- > 0;)
    {
        strides[k] = stride;
        stride *= shape[k];
    }
    return strides;
}

size_t span_in_elements(const Shape& shape, const std::vector<size_t>& strides)
{
    assert(shape.rank() == strides.size());

    if (shape.size() == 0)
    {
        return 0; // a tensor with a zero dimension reaches nothing
    }

    // The largest reachable offset uses the last valid index of every dimension.
    size_t last = 0;
    for (size_t k = 0; k < shape.rank(); ++k)
    {
        last += (shape[k] - 1) * strides[k];
    }
    return last + 1;
}

std::string to_string(const std::vector<size_t>& strides)
{
    std::string out = "(";
    for (size_t k = 0; k < strides.size(); ++k)
    {
        if (k > 0)
        {
            out += ", ";
        }
        out += std::to_string(strides[k]);
    }
    out += ")";
    return out;
}

} // namespace veda::core
