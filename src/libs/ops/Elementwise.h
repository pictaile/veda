#ifndef VEDA_ELEMENTWISE_H
#define VEDA_ELEMENTWISE_H

#include "Tensor.h"

namespace veda::ops
{

// Element-wise operations over broadcast operands.
//
// Every op in this layer is a pure function: inputs in, a freshly allocated contiguous tensor out,
// nothing mutated. That is what makes each one testable against hand-computed numbers, and it is
// the shape an autograd tape needs later (architecture §2 and §7) — an in-place variant added for
// speed would have to be unpicked before training is possible.
//
// Inputs may be views of any kind, including non-contiguous ones and views over the same storage:
// add(x, x) is legal, and so is add(x, x.transpose(0, 1)).

core::Tensor add(const core::Tensor& a, const core::Tensor& b);
core::Tensor mul(const core::Tensor& a, const core::Tensor& b);

core::Tensor add(const core::Tensor& a, float scalar);
core::Tensor mul(const core::Tensor& a, float scalar);

} // namespace veda::ops

#endif //VEDA_ELEMENTWISE_H
