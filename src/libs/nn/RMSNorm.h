#ifndef VEDA_RMSNORM_H
#define VEDA_RMSNORM_H

#include "Tensor.h"

namespace veda::nn
{

// Normalisation with a learned per-channel scale:
//
//     out = gamma * (x / sqrt(mean(x^2) + eps))
//
// The clearest case of the op/layer split (architecture §2): ops::rms_normalize is the maths and
// owns nothing, which is why it could be verified on numbers computed by hand two epics before any
// model file existed. gamma is a weight, and a weight makes this a layer.
//
// The bare op fixes every vector's magnitude and keeps its direction; gamma then re-weights each
// channel — a controlled amount of the discarded magnitude information handed back. It cannot
// reintroduce the compounding the norm exists to prevent, because its input always has RMS 1.
//
// Called 57 times per token in Qwen3-0.6B: twice per block, plus the final norm before the LM head.
// AD2: gamma is a view into the loader's buffer, never allocated here.
class RMSNorm
{
public:
    // eps comes from the model config (E4.S3.T6), never from a constant here — there is no
    // sensible default for it.
    RMSNorm(core::Tensor gamma, float eps);

    // [..., D] -> [..., D]. gamma applies per channel across every leading dimension: every token
    // in the sequence meets the same vector.
    core::Tensor forward(const core::Tensor& x) const;

    size_t hidden_size() const { return gamma_.shape()[0]; }
    float eps() const noexcept { return eps_; }
    const core::Tensor& gamma() const noexcept { return gamma_; }

private:
    core::Tensor gamma_;
    float eps_;
};

} // namespace veda::nn

#endif //VEDA_RMSNORM_H
