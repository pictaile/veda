#ifndef VEDA_WALK_H
#define VEDA_WALK_H

#include "Shape.h"
#include "Storage.h"
#include "Tensor.h"

#include <cstddef>
#include <vector>

// Internal to veda::ops: the index walk every op in this layer shares.
//
// After E1 an operand is rarely contiguous — it may be transposed, sliced, or broadcast with a
// stride of zero — so ops address their inputs by coordinates and the offset formula rather than
// by running along a buffer. This header exists so that walk is written once.
namespace veda::ops::detail
{

// Advances an index in row-major order: last dimension fastest, carrying leftwards.
inline void advance(std::vector<size_t>& index, const core::Shape& shape)
{
    for (size_t k = shape.rank(); k-- > 0;)
    {
        if (++index[k] < shape[k])
        {
            return;
        }
        index[k] = 0;
    }
}

// The same, skipping one axis — used when that axis is what a lane runs along.
inline void advance_skipping(std::vector<size_t>& index, const core::Shape& shape, size_t skipped)
{
    for (size_t k = shape.rank(); k-- > 0;)
    {
        if (k == skipped)
        {
            continue;
        }
        if (++index[k] < shape[k])
        {
            return;
        }
        index[k] = 0;
    }
}

inline size_t offset_of(const std::vector<size_t>& strides, const std::vector<size_t>& index)
{
    size_t offset = 0;
    for (size_t k = 0; k < strides.size(); ++k)
    {
        offset += index[k] * strides[k];
    }
    return offset;
}

inline float read(const core::Tensor& tensor, const std::vector<size_t>& index)
{
    return tensor.storage()->data()[tensor.offset() + offset_of(tensor.strides(), index)];
}

// Applies a function to every element, into a freshly allocated contiguous result. The output is
// filled straight through; the input is walked through its strides.
template <typename Op>
core::Tensor map_elements(const core::Tensor& tensor, Op op)
{
    core::Tensor out{tensor.shape()};
    if (tensor.numel() == 0)
    {
        return out;
    }

    std::vector<size_t> index(tensor.rank(), 0);
    float* destination = out.data();
    for (size_t i = 0; i < tensor.numel(); ++i)
    {
        destination[i] = op(read(tensor, index));
        advance(index, tensor.shape());
    }
    return out;
}

} // namespace veda::ops::detail

#endif //VEDA_WALK_H
