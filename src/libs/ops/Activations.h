#ifndef VEDA_ACTIVATIONS_H
#define VEDA_ACTIVATIONS_H

#include "Tensor.h"

namespace veda::ops
{

// silu(x) = x / (1 + exp(-x))
//
// The non-linearity Qwen3's feed-forward network applies to its gate branch, in all 28 layers.
// Without some non-linearity between projections a deep stack collapses: W2(W1 x) is just (W2 W1) x,
// and depth buys nothing.
//
// Approaches x for large positive inputs and 0 for large negative ones, smoothly and without a
// branch — at x = -100, exp(100) is inf and x/inf is -0, which is the right answer.
core::Tensor silu(const core::Tensor& tensor);

// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// The tanh approximation — what PyTorch calls gelu(approximate='tanh') and what most published
// models were trained against. The exact form uses erf and differs in the fourth decimal. Qwen3
// does not use GELU at all; it is here because "which GELU" is exactly the kind of unstated detail
// that costs an afternoon later, so the choice is recorded rather than left implicit.
core::Tensor gelu(const core::Tensor& tensor);

} // namespace veda::ops

#endif //VEDA_ACTIVATIONS_H
