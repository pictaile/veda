#ifndef VEDA_BROADCAST_H
#define VEDA_BROADCAST_H

#include "Shape.h"
#include "Tensor.h"

namespace veda::ops
{

// The shape two operands combine to.
//
// Shapes are aligned from the right; each pair of extents must be equal or contain a 1, and a
// missing leading dimension counts as 1. Throws std::invalid_argument, naming both shapes, when
// they do not broadcast.
core::Shape broadcast_shape(const core::Shape& a, const core::Shape& b);

// Reads a tensor as if it had the target shape, by putting a stride of zero on every axis that is
// repeated. Advancing an axis of stride zero skips nothing, so every index along it lands on the
// same element — repetition expressed in the description, with no data copied.
//
// The result aliases its input and must never be written through: several logical positions map
// to one physical element. Throws std::invalid_argument if the target is not reachable.
core::Tensor broadcast_to(const core::Tensor& tensor, const core::Shape& target);

} // namespace veda::ops

#endif //VEDA_BROADCAST_H
