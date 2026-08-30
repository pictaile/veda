#include "Linear.h"

#include "Elementwise.h"
#include "Matmul.h"
#include "Shape.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace veda::nn
{

using core::Tensor;

namespace
{
void require_rank_2(const Tensor& weight)
{
    if (weight.rank() != 2)
    {
        throw std::invalid_argument("nn::Linear: the weight must be rank 2 [out, in], got " +
                                    weight.shape().to_string());
    }
}
} // namespace

Linear::Linear(Tensor weight) : weight_(std::move(weight))
{
    require_rank_2(weight_);
}

Linear::Linear(Tensor weight, Tensor bias) : weight_(std::move(weight)), bias_(std::move(bias))
{
    require_rank_2(weight_);

    if (bias_->rank() != 1 || bias_->shape()[0] != out_features())
    {
        throw std::invalid_argument("nn::Linear: the bias must be [" +
                                    std::to_string(out_features()) + "], got " +
                                    bias_->shape().to_string());
    }
}

Tensor Linear::forward(const Tensor& x) const
{
    if (x.rank() < 2 || x.shape()[x.rank() - 1] != in_features())
    {
        throw std::invalid_argument("nn::Linear: input " + x.shape().to_string() +
                                    " does not end in in_features " +
                                    std::to_string(in_features()));
    }

    // matmul_nt carries the [out, in] layout (AD4) and iterates the batch dimensions; add
    // broadcasts the [out] bias across every leading dimension. There is no loop here, and
    // nowhere for a layout bug to hide.
    Tensor y = ops::matmul_nt(x, weight_);
    return bias_ ? ops::add(y, *bias_) : y;
}

} // namespace veda::nn
