#include "FeedForward.h"

#include "Activations.h"
#include "Elementwise.h"
#include "Profile.h"
#include "Shape.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace veda::model
{

using core::Tensor;

FeedForward::FeedForward(Weights weights) : weights_(std::move(weights))
{
    // A weight file whose up_proj is a different width from its gate_proj is broken, and saying so
    // at construction is worth more than a matmul error seven layers into a forward pass.
    if (weights_.gate_proj.out_features() != weights_.up_proj.out_features())
    {
        throw std::invalid_argument("model::FeedForward: gate_proj produces " +
                                    std::to_string(weights_.gate_proj.out_features()) +
                                    " but up_proj produces " +
                                    std::to_string(weights_.up_proj.out_features()));
    }
    if (weights_.gate_proj.in_features() != weights_.up_proj.in_features())
    {
        throw std::invalid_argument("model::FeedForward: gate_proj and up_proj take different "
                                    "input widths");
    }
    if (weights_.down_proj.in_features() != weights_.gate_proj.out_features())
    {
        throw std::invalid_argument("model::FeedForward: down_proj takes " +
                                    std::to_string(weights_.down_proj.in_features()) +
                                    ", the inner width is " +
                                    std::to_string(weights_.gate_proj.out_features()));
    }
    if (weights_.down_proj.out_features() != weights_.gate_proj.in_features())
    {
        throw std::invalid_argument("model::FeedForward: down_proj produces " +
                                    std::to_string(weights_.down_proj.out_features()) +
                                    ", the hidden width is " +
                                    std::to_string(weights_.gate_proj.in_features()));
    }
}

Tensor FeedForward::forward(const Tensor& x) const
{
    const profile::Scope scope("feed_forward");
    // silu on the gate branch only — that asymmetry is the gate. Left as two composed ops rather
    // than a fused swiglu: fusing halves the traffic through the [B, T, F] intermediate and hides
    // the structure, and E13 measures before deciding.
    const Tensor gate = ops::silu(weights_.gate_proj.forward(x));
    const Tensor up = weights_.up_proj.forward(x);
    return weights_.down_proj.forward(ops::mul(gate, up));
}

} // namespace veda::model
