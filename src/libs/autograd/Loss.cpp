#include "Loss.h"

#include "Shape.h"
#include "Softmax.h"
#include "Storage.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace veda::autograd
{

using core::Shape;
using core::Tensor;

Variable cross_entropy(const Variable& logits, const std::vector<int64_t>& targets)
{
    const Tensor& scores = logits.value();
    if (scores.rank() == 0)
    {
        throw std::invalid_argument("cross_entropy: logits must have at least one dimension");
    }

    const size_t vocabulary = scores.shape()[scores.rank() - 1];
    const size_t lanes = vocabulary == 0 ? 0 : scores.numel() / vocabulary;

    if (targets.size() != lanes)
    {
        throw std::invalid_argument(
            "cross_entropy: " + std::to_string(targets.size()) + " targets for " +
            std::to_string(lanes) + " lanes of shape " + scores.shape().to_string());
    }
    if (lanes == 0)
    {
        throw std::invalid_argument("cross_entropy: nothing to average over");
    }
    for (size_t lane = 0; lane < lanes; ++lane)
    {
        if (targets[lane] < 0 || static_cast<size_t>(targets[lane]) >= vocabulary)
        {
            throw std::invalid_argument(
                "cross_entropy: target " + std::to_string(targets[lane]) + " at lane " +
                std::to_string(lane) + " is outside [0, " + std::to_string(vocabulary) + ")");
        }
    }

    // Log space throughout the forward: -log p_t read straight off, never divided into existence.
    const Tensor log_probabilities = ops::log_softmax(scores, scores.rank() - 1);

    double total = 0.0;
    for (size_t lane = 0; lane < lanes; ++lane)
    {
        const size_t at = lane * vocabulary + static_cast<size_t>(targets[lane]);
        total += -static_cast<double>(log_probabilities.data()[at]);
    }

    Tensor loss{Shape({})};
    loss.data()[0] = static_cast<float>(total / static_cast<double>(lanes));

    if (!logits.requires_grad())
    {
        return Variable::leaf(std::move(loss), false);
    }

    // The node keeps the probabilities, not the logits: p is what the backward needs, and it is
    // the same size. exp of a log-probability is in [0, 1] and cannot overflow.
    Tensor probabilities{log_probabilities.shape()};
    for (size_t i = 0; i < probabilities.numel(); ++i)
    {
        probabilities.data()[i] = std::exp(log_probabilities.data()[i]);
    }

    auto input = logits.node();
    auto node = std::make_shared<Node>(std::move(loss));
    node->requires_grad = true;
    node->parents = {input};
    node->backward = [input, probabilities, targets, vocabulary, lanes](const Tensor& g) {
        // dL/dx = (p - one_hot) / lanes. A subtraction, because the softmax's dense Jacobian and
        // the log's reciprocal cancel; and divided by the lane count to match the mean in the
        // forward — dividing in one place only gives a gradient of exactly the right shape and
        // `lanes` times too large, which nothing but finite differences would ever report.
        const float seed = g.data()[0] / static_cast<float>(lanes);

        Tensor grad{probabilities.shape()};
        for (size_t i = 0; i < grad.numel(); ++i)
        {
            grad.data()[i] = seed * probabilities.data()[i];
        }
        for (size_t lane = 0; lane < lanes; ++lane)
        {
            grad.data()[lane * vocabulary + static_cast<size_t>(targets[lane])] -= seed;
        }

        for (size_t i = 0; i < grad.numel(); ++i)
        {
            input->grad.data()[i] += grad.data()[i];
        }
    };

    return Variable(std::move(node));
}

} // namespace veda::autograd
