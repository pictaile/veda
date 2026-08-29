#ifndef VEDA_NORMALIZE_H
#define VEDA_NORMALIZE_H

#include "Tensor.h"

namespace veda::ops
{

// Root-mean-square normalisation, along the last dimension:
//
//     rms = sqrt( (1/n) * sum of xi^2  +  eps )
//     out = x / rms
//
// Keeps the direction of each vector and fixes its magnitude — the output's mean square is 1. That
// is what stops activation magnitudes compounding across 28 layers.
//
// LayerNorm additionally subtracts the mean; dropping it is the whole of RMSNorm, and it costs
// nothing in quality for transformers while saving a pass over the vector.
//
// The axis is fixed rather than a parameter: every use in the model normalises a token's hidden
// vector across D, which is always the last dimension, and an axis chosen wrongly produces
// plausible numbers no test would catch.
//
// No learned scale here: the model's layer is gamma * rms_normalize(x), and gamma is a weight,
// which makes it nn::RMSNorm (E6). Keeping the maths weight-free is what lets it be verified on
// numbers computed by hand.
//
// eps comes from the model config (1e-6 for Qwen3), never from a constant in the code. Throws
// std::invalid_argument on a rank-0 input, which has no last dimension.
core::Tensor rms_normalize(const core::Tensor& tensor, float eps);

} // namespace veda::ops

#endif //VEDA_NORMALIZE_H
