#ifndef VEDA_STRIDES_H
#define VEDA_STRIDES_H

#include "Shape.h"

#include <cstddef>
#include <string>
#include <vector>

namespace veda::core
{

// How many elements to skip in the flat buffer to advance each dimension by one.
//
// Memory is one-dimensional, a tensor is not. Strides are the rule that turns coordinates into
// a flat index:
//
//     offset = sum over k of  i_k * s_k
//
// Row-major and packed means the last dimension steps by 1, and every dimension to its left
// steps over a whole block of everything to its right:
//
//     (2, 3)     -> (3, 1)
//     (2, 3, 4)  -> (12, 4, 1)
//
// Strides are counted in elements, never bytes.
std::vector<size_t> contiguous_strides(const Shape& shape);

// The number of elements a tensor spans from its own start, i.e. one past the largest reachable
// offset. Equal to shape.size() while the strides are contiguous, larger once a view is strided.
size_t span_in_elements(const Shape& shape, const std::vector<size_t>& strides);

std::string to_string(const std::vector<size_t>& strides);

} // namespace veda::core

#endif //VEDA_STRIDES_H
