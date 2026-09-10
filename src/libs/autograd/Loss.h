#ifndef VEDA_LOSS_H
#define VEDA_LOSS_H

#include "Shape.h"
#include "Tensor.h"
#include "Variable.h"

#include <cstdint>
#include <vector>

// The number the whole training apparatus tries to make smaller.
//
// Everything before this computed a distribution; nothing said whether it was any good.
// Cross-entropy is the surprise of the model's claim about the token that actually came next:
//
//     L = -log p_t
//
// Zero when certain and right, unbounded when certain and wrong, log V for a uniform guess — 11.93
// at Qwen3's vocabulary, which is where an untrained run starts and the first sanity check of any
// training loop.
namespace veda::autograd
{

// logits [.., V], one target id per lane, the result a scalar: the mean over lanes.
//
// FUSED, and not for speed. Written as three ops — softmax, log, pick — the middle one produces
// -inf on ordinary inputs (softmax underflows a small probability to exactly 0) and the backward
// is a product with a dense Jacobian. Written as one, the loss stays in log space and the whole
// gradient collapses to
//
//     dL/dx_i = p_i - [i == target]
//
// The softmax's Jacobian and the log's reciprocal cancel exactly. Read as a statement about
// learning: THE GRADIENT IS THE ERROR — push down what was predicted, push up what happened, in
// proportion to how wrong it was. A confident correct prediction contributes nearly nothing; a
// confident wrong one contributes nearly 1.
//
// The mean rather than the sum, so that the loss and therefore the step size do not depend on how
// many positions are in the batch.
//
// Throws std::invalid_argument if the target count does not match the lane count, or a target is
// outside [0, V). No gradient flows to the targets; they are ids.
Variable cross_entropy(const Variable& logits, const std::vector<int64_t>& targets);

} // namespace veda::autograd

#endif //VEDA_LOSS_H
