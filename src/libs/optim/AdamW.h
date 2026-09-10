#ifndef VEDA_ADAMW_H
#define VEDA_ADAMW_H

#include "Tensor.h"
#include "Variable.h"

#include <cstddef>
#include <vector>

// The rule that turns a gradient into a change in the weights.
//
// The gradient says which way is downhill RIGHT HERE, and plain gradient descent believes it
// completely. Two problems follow: the locally steepest direction is rarely the useful one, and a
// step size that suits one parameter is wrong for another by orders of magnitude. Adam answers both
// with two running averages; AdamW then fixes what Adam got wrong about weight decay.
//
// This is also where training acquires STATE. Everything before was a function of its inputs; an
// optimiser remembers — two extra tensors per parameter, which is why AdamW costs twice what the
// weights do and dominates the memory arithmetic of any training run.
namespace veda::optim
{

struct AdamWConfig
{
    float learning_rate = 1e-3f;
    float beta1 = 0.9f;    // momentum: roughly the last ten gradients
    float beta2 = 0.999f;  // the scale: roughly the last thousand squared gradients
    float eps = 1e-8f;     // outside the square root — the reference convention (AD7)
    float weight_decay = 0.01f;
};

class AdamW
{
public:
    explicit AdamW(std::vector<autograd::Variable> parameters, AdamWConfig config = {});

    // theta -= lr . m_hat / (sqrt(v_hat) + eps)      the adaptive part
    // theta -= lr . weight_decay . theta             the decay, NOT divided by sqrt(v)
    //
    // The decoupling is the whole point of the W. Adam with L2 adds lambda.theta to the gradient,
    // so the decay is then divided by sqrt(v) along with everything else — and a parameter with
    // large gradients ends up LESS regularised than one with small gradients. Nobody intends that.
    //
    // Does not clear the gradients: accumulating over micro-batches is why step() and zero_grad()
    // are separate operations everywhere.
    void step();

    void zero_grad();

    size_t steps() const noexcept { return steps_; }
    const std::vector<autograd::Variable>& parameters() const noexcept { return parameters_; }

private:
    std::vector<autograd::Variable> parameters_;
    std::vector<core::Tensor> first_moment_;
    std::vector<core::Tensor> second_moment_;
    AdamWConfig config_;
    size_t steps_ = 0;
};

} // namespace veda::optim

#endif //VEDA_ADAMW_H
