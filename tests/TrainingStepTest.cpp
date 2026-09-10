// E15 milestone — a parameter set moves downhill on a fixed batch, and the loss falls.
//
// The first place where every piece of the training extension runs together in the order a real
// loop uses it:
//
//     forward -> loss -> backward -> clip -> step -> zero_grad
//
// Nothing here is a mock. The tape is E14's, the loss is E15.S1's, the optimiser E15.S2's, and the
// forward pass is built from the same veda::ops the inference path uses.

#include "AdamW.h"
#include "Clip.h"
#include "Loss.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"
#include "Variable.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using veda::autograd::cross_entropy;
using veda::autograd::matmul_nt;
using veda::autograd::silu;
using veda::autograd::Variable;
using veda::core::Shape;
using veda::core::Tensor;
using veda::optim::AdamW;
using veda::optim::AdamWConfig;
using veda::optim::clip_gradient_norm;

namespace
{
Tensor random_tensor(const Shape& shape, std::mt19937& engine, float spread)
{
    std::uniform_real_distribution<float> values(-spread, spread);
    Tensor t{shape};
    for (size_t i = 0; i < shape.size(); ++i)
    {
        t.data()[i] = values(engine);
    }
    return t;
}
} // namespace

int main()
{
    std::mt19937 engine(20260912);

    const size_t features = 6;
    const size_t hidden = 12;
    const size_t classes = 4;
    const size_t batch = 8;

    // One fixed batch, and a network with no chance of generalising — it can only memorise, which
    // is exactly the property being tested: if the machinery works, the loss goes to nearly zero.
    const Tensor inputs = random_tensor(Shape({batch, features}), engine, 1.0f);
    std::vector<int64_t> targets(batch);
    for (size_t i = 0; i < batch; ++i)
    {
        targets[i] = static_cast<int64_t>(i % classes);
    }

    Variable w1 = Variable::leaf(random_tensor(Shape({hidden, features}), engine, 0.4f));
    Variable w2 = Variable::leaf(random_tensor(Shape({classes, hidden}), engine, 0.4f));
    const Variable x = Variable::leaf(inputs, false);

    AdamW optimiser({w1, w2}, AdamWConfig{0.05f, 0.9f, 0.999f, 1e-8f, 0.0f});

    auto loss_now = [&] {
        return cross_entropy(matmul_nt(silu(matmul_nt(x, w1)), w2), targets).value().at({});
    };

    const double before = loss_now();

    // *** an untrained model sits at log(classes) — the first sanity check of any training run ***
    CHECK_NEAR(before, std::log(static_cast<double>(classes)), 0.35);

    double largest_norm = 0.0;
    for (int step = 0; step < 300; ++step)
    {
        optimiser.zero_grad();

        Variable hidden_activations = silu(matmul_nt(x, w1));
        Variable logits = matmul_nt(hidden_activations, w2);
        Variable loss = cross_entropy(logits, targets);

        loss.backward();

        // after the backward — the gradients must exist; before the step — afterwards the
        // parameters have already moved
        const float norm = clip_gradient_norm({w1, w2}, 1.0f);
        largest_norm = norm > largest_norm ? norm : largest_norm;

        optimiser.step();
    }

    const double after = loss_now();

    // *** the loss fell ***
    CHECK(after < before);
    CHECK(after < 0.01);
    CHECK_EQ(optimiser.steps(), size_t{300});

    // and the model now predicts every target of the fixed batch
    {
        Variable prediction = matmul_nt(silu(matmul_nt(x, w1)), w2);
        for (size_t row = 0; row < batch; ++row)
        {
            size_t best = 0;
            for (size_t c = 1; c < classes; ++c)
            {
                if (prediction.value().at({row, c}) > prediction.value().at({row, best}))
                {
                    best = c;
                }
            }
            CHECK_EQ(static_cast<int64_t>(best), targets[row]);
        }
    }

    // Clipping never fired at max_norm = 1: cross-entropy's gradient is p - one_hot, bounded by 1
    // per element by construction, so a well-behaved run like this one stays under the threshold.
    // That is the guard doing its job — it exists for the batch that misbehaves, not this one.
    CHECK(largest_norm > 0.0);
    CHECK(std::isfinite(largest_norm));
    CHECK(largest_norm <= 1.0);

    // and it does engage when the norm is genuinely exceeded, on this very network
    {
        optimiser.zero_grad();
        cross_entropy(matmul_nt(silu(matmul_nt(x, w1)), w2), targets).backward();

        const float threshold = 1e-4f;
        const float reported = clip_gradient_norm({w1, w2}, threshold);
        if (reported > threshold)
        {
            double squares = 0.0;
            for (const Variable& p : {w1, w2})
            {
                for (size_t i = 0; i < p.grad().numel(); ++i)
                {
                    squares += static_cast<double>(p.grad().data()[i]) * p.grad().data()[i];
                }
            }
            CHECK_NEAR(std::sqrt(squares), threshold, 1e-6);
        }
        optimiser.zero_grad();
    }

    // the same run with a broken gradient must NOT converge — the milestone has to be able to fail
    {
        std::mt19937 other(20260912);
        const Tensor same_inputs = random_tensor(Shape({batch, features}), other, 1.0f);
        Variable b1 = Variable::leaf(random_tensor(Shape({hidden, features}), other, 0.4f));
        Variable b2 = Variable::leaf(random_tensor(Shape({classes, hidden}), other, 0.4f));
        const Variable bx = Variable::leaf(same_inputs, false);

        AdamW broken({b1, b2}, AdamWConfig{0.05f, 0.9f, 0.999f, 1e-8f, 0.0f});
        for (int step = 0; step < 300; ++step)
        {
            broken.zero_grad();
            cross_entropy(matmul_nt(silu(matmul_nt(bx, b1)), b2), targets).backward();

            // the sign flipped: the step now climbs
            for (const Variable& p : {b1, b2})
            {
                for (size_t i = 0; i < p.grad().numel(); ++i)
                {
                    p.node()->grad.data()[i] = -p.node()->grad.data()[i];
                }
            }
            broken.step();
        }

        const double uphill =
            cross_entropy(matmul_nt(silu(matmul_nt(bx, b1)), b2), targets).value().at({});
        CHECK(uphill > before);
    }

    return VEDA_TEST_SUMMARY("TrainingStepTest");
}
