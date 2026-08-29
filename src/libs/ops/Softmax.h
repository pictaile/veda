#ifndef VEDA_SOFTMAX_H
#define VEDA_SOFTMAX_H

#include "Tensor.h"

#include <cstddef>

namespace veda::ops
{

// Turns arbitrary scores into a distribution, along one dimension:
//
//     softmax(x)i = exp(xi - m) / sum over j of exp(xj - m),      m = max over the lane
//
// The subtraction of the maximum is not an optimisation. exp overflows fp32 just past 88, and
// attention scores in the hundreds are ordinary; without it the result is inf/inf = NaN. Softmax
// is shift-invariant, so subtracting m computes the same function with the largest argument at
// exactly 0 — overflow impossible, and a denominator of at least 1.
//
// Each lane is a one-dimensional run along `dim` with every other coordinate fixed: [B,H,T,T] with
// dim = 3 has B*H*T lanes of length T. A -inf entry (a masked position, E7) comes out as exactly 0.
//
// Throws std::invalid_argument if dim is not below the rank. The result is freshly allocated and
// contiguous; the input is untouched and may be non-contiguous.
core::Tensor softmax(const core::Tensor& tensor, size_t dim);

} // namespace veda::ops

#endif //VEDA_SOFTMAX_H
