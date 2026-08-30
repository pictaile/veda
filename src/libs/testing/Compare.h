#ifndef VEDA_COMPARE_H
#define VEDA_COMPARE_H

#include "Tensor.h"

#include <cstddef>
#include <string>
#include <vector>

// The instrument every later epic is debugged with.
//
// Hand-computed 2x2 tests catch algebra errors. They do not catch a transposed weight, a swapped
// head layout or a wrong RoPE base — those compute without error and return confident nonsense
// (AD7). From E6 onward, "is this right?" is answered by comparing a tensor against a reference
// value, and the answer is only useful if the failure says where.
//
// Development tooling: nothing in the veda binary links against this.
namespace veda::testing
{

// Where and how badly two tensors disagree.
struct Difference
{
    bool equal = true;
    std::vector<size_t> index;   // coordinates of the worst offender
    float actual = 0.0f;
    float expected = 0.0f;
    float absolute = 0.0f;
    float relative = 0.0f;
    size_t mismatches = 0;       // how many elements failed in total
    size_t total = 0;
    std::string message;
};

// Compares element by element under the usual rule:
//
//     |actual - expected| <= atol + rtol * |expected|
//
// Non-finite values are settled before that rule is applied, because it gets them backwards: it
// would call +inf and -inf equal (both sides infinite) and two equal infinities different
// (inf - inf is NaN). Equal infinities pass, opposite ones fail, and any NaN fails — a NaN in an
// activation is always a bug.
//
// Reports the *worst* offender, not the first: the first is an accident of iteration order, the
// worst is where the disagreement is most structural. The mismatch count is what separates "one
// element is wrong" (an indexing bug) from "everything is wrong" (a transposed operand).
//
// Different shapes are reported as such, without comparing values. Both operands may be
// non-contiguous.
Difference compare_close(const core::Tensor& actual, const core::Tensor& expected, float rtol,
                         float atol);

// The same comparison, throwing std::runtime_error with the message on failure.
void assert_close(const core::Tensor& actual, const core::Tensor& expected, float rtol, float atol);

} // namespace veda::testing

#endif //VEDA_COMPARE_H
