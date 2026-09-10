#ifndef VEDA_CLIP_H
#define VEDA_CLIP_H

#include "Variable.h"

#include <vector>

namespace veda::optim
{

// A ceiling on how far one step may move, applied between the backward pass and the optimiser.
//
// Training is stable until one batch is not: an unusual example, a small normaliser, a long
// dependency, and a gradient arrives an order of magnitude larger than usual. Adam does not save
// you — it adapts over beta2 = 0.999, roughly a thousand steps, so a single spike passes through
// nearly unmodified and lands in the weights. One bad step can undo hours.
//
// GLOBAL, not per tensor:
//
//     total = sqrt(sum over every parameter, every element, of g^2)
//     if total > max_norm:  every g *= max_norm / total
//
// The word that matters is EVERY. One scale factor across all gradients leaves their ratios intact,
// so the update points exactly where it did, just less far. Clipping each tensor against its own
// threshold would rescale them by different factors and turn the step toward a direction no
// gradient ever pointed in. Clipping shortens the step; it must not turn it.
//
// Returns the norm BEFORE clipping — the most informative single number in a training log. Its
// usual size tells you what max_norm should be, a jump identifies the batch that caused it, and a
// run that clips every step has quietly become sign descent.
//
// A non-finite total is returned as-is and nothing is scaled: multiplying every parameter by NaN
// would spread one bad gradient across the entire model.
float clip_gradient_norm(const std::vector<autograd::Variable>& parameters, float max_norm);

} // namespace veda::optim

#endif //VEDA_CLIP_H
