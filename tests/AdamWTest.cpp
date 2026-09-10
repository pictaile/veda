// E15.S2.T2 — AdamW: momentum, the second moment, and the decoupled decay

#include "AdamW.h"
#include "Loss.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"
#include "Variable.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

using veda::autograd::Variable;
using veda::core::Shape;
using veda::core::Tensor;
using veda::optim::AdamW;
using veda::optim::AdamWConfig;

namespace
{
Tensor filled(const Shape& shape, const std::vector<float>& values)
{
    Tensor t{shape};
    for (size_t i = 0; i < values.size(); ++i)
    {
        t.data()[i] = values[i];
    }
    return t;
}

// Writes a gradient directly, so a step can be tested without a forward pass in the way.
void set_grad(const Variable& v, const std::vector<float>& values)
{
    for (size_t i = 0; i < values.size(); ++i)
    {
        v.node()->grad.data()[i] = values[i];
    }
}
} // namespace

int main()
{
    // --- one step by hand ---------------------------------------------------------------------------
    {
        // theta = 1, g = 2, lr = 0.1, no decay
        //   m = 0.2, v = 0.008;  m_hat = 2, v_hat = 4;  update = 2/2 = 1;  theta = 0.9
        Variable theta = Variable::leaf(filled(Shape({1}), {1.0f}));
        AdamW optimiser({theta}, AdamWConfig{0.1f, 0.9f, 0.999f, 1e-8f, 0.0f});

        set_grad(theta, {2.0f});
        optimiser.step();

        CHECK_NEAR(theta.value().at({0}), 0.9, 1e-5);
        CHECK_EQ(optimiser.steps(), size_t{1});

        // *** step() does not clear the gradient — accumulation over micro-batches is why ***
        CHECK_EQ(theta.grad().at({0}), 2.0f);
        optimiser.zero_grad();
        CHECK_EQ(theta.grad().at({0}), 0.0f);
        CHECK_EQ(optimiser.steps(), size_t{1});   // zero_grad does not touch the counter
    }

    // *** the first step is the learning rate, whatever the gradient's magnitude ***
    {
        Variable tiny = Variable::leaf(filled(Shape({1}), {0.0f}));
        Variable huge = Variable::leaf(filled(Shape({1}), {0.0f}));
        AdamW optimiser({tiny, huge}, AdamWConfig{0.1f, 0.9f, 0.999f, 1e-8f, 0.0f});

        set_grad(tiny, {0.001f});     // gradients a million apart
        set_grad(huge, {1000.0f});
        optimiser.step();

        CHECK_NEAR(tiny.value().at({0}), -0.1, 1e-5);
        CHECK_NEAR(huge.value().at({0}), -0.1, 1e-5);
        CHECK_NEAR(tiny.value().at({0}), huge.value().at({0}), 1e-6);

        // and the sign still follows the gradient
        Variable negative = Variable::leaf(filled(Shape({1}), {0.0f}));
        AdamW other({negative}, AdamWConfig{0.1f, 0.9f, 0.999f, 1e-8f, 0.0f});
        set_grad(negative, {-5.0f});
        other.step();
        CHECK_NEAR(negative.value().at({0}), 0.1, 1e-5);
    }

    // *** bias correction: without it the first step would be 3.16x too large ***
    {
        // uncorrected: 0.1g / sqrt(0.001 g^2) = 3.1623 . sign(g)
        Variable theta = Variable::leaf(filled(Shape({1}), {0.0f}));
        AdamW optimiser({theta}, AdamWConfig{0.1f, 0.9f, 0.999f, 1e-8f, 0.0f});
        set_grad(theta, {1.0f});
        optimiser.step();

        const float moved = std::fabs(theta.value().at({0}));
        CHECK_NEAR(moved, 0.1, 1e-5);                  // corrected
        CHECK(std::fabs(moved - 0.31623f) > 1e-2f);    // and not the uncorrected value
    }

    // --- weight decay, with nothing else in the way --------------------------------------------------
    {
        // g = 0 makes every adaptive term vanish; only the decay is left, so the expected value is
        // arithmetic rather than a simulation: theta(1 - lr . lambda) per step
        Variable theta = Variable::leaf(filled(Shape({1}), {1.0f}));
        AdamW optimiser({theta}, AdamWConfig{0.1f, 0.9f, 0.999f, 1e-8f, 0.01f});

        set_grad(theta, {0.0f});
        optimiser.step();
        CHECK_NEAR(theta.value().at({0}), 0.999, 1e-6);

        optimiser.step();
        CHECK_NEAR(theta.value().at({0}), 0.999 * 0.999, 1e-6);
    }

    // *** decoupling: equal values with unequal gradients decay by equal amounts ***
    {
        const AdamWConfig with_decay{0.1f, 0.9f, 0.999f, 1e-8f, 0.05f};
        const AdamWConfig without_decay{0.1f, 0.9f, 0.999f, 1e-8f, 0.0f};

        Variable a = Variable::leaf(filled(Shape({1}), {2.0f}));
        Variable b = Variable::leaf(filled(Shape({1}), {2.0f}));
        AdamW decayed({a, b}, with_decay);
        set_grad(a, {0.001f});
        set_grad(b, {500.0f});
        decayed.step();

        Variable a_plain = Variable::leaf(filled(Shape({1}), {2.0f}));
        Variable b_plain = Variable::leaf(filled(Shape({1}), {2.0f}));
        AdamW plain({a_plain, b_plain}, without_decay);
        set_grad(a_plain, {0.001f});
        set_grad(b_plain, {500.0f});
        plain.step();

        // The decay contribution is the difference between the two runs, and it is the same for
        // both parameters despite gradients half a million apart — which is precisely what Adam
        // with L2 would fail to do, since there the decay is divided by sqrt(v) as well.
        const double decay_a = a_plain.value().at({0}) - a.value().at({0});
        const double decay_b = b_plain.value().at({0}) - b.value().at({0});
        CHECK_NEAR(decay_a, 0.1 * 0.05 * 2.0, 1e-6);
        CHECK_NEAR(decay_a, decay_b, 1e-6);
    }

    // --- a quadratic converges -----------------------------------------------------------------------
    {
        // f(x) = sum (x - target)^2, gradient 2(x - target) written by hand
        const std::vector<float> target = {3.0f, -1.0f, 0.5f, 7.0f};
        Variable x = Variable::leaf(Tensor{Shape({4})});
        AdamW optimiser({x}, AdamWConfig{0.1f, 0.9f, 0.999f, 1e-8f, 0.0f});

        double first = 0.0;
        double last = 0.0;
        for (int step = 0; step < 400; ++step)
        {
            double loss = 0.0;
            std::vector<float> gradient(4);
            for (size_t i = 0; i < 4; ++i)
            {
                const double error = x.value().at({i}) - target[i];
                loss += error * error;
                gradient[i] = static_cast<float>(2.0 * error);
            }
            if (step == 0)
            {
                first = loss;
            }
            last = loss;

            optimiser.zero_grad();
            set_grad(x, gradient);
            optimiser.step();
        }

        CHECK(first > 50.0);
        CHECK(last < 1e-4);
        for (size_t i = 0; i < 4; ++i)
        {
            CHECK_NEAR(x.value().at({i}), target[i], 1e-2);
        }
        CHECK_EQ(optimiser.steps(), size_t{400});
    }

    // --- through the tape, with a real loss ------------------------------------------------------------
    {
        // a linear model over four classes, one fixed example: the loss must fall
        std::mt19937 engine(20260912);
        std::uniform_real_distribution<float> spread(-0.5f, 0.5f);

        Tensor start{Shape({1, 4})};
        for (size_t i = 0; i < 4; ++i)
        {
            start.data()[i] = spread(engine);
        }

        Variable logits = Variable::leaf(start);
        AdamW optimiser({logits}, AdamWConfig{0.1f, 0.9f, 0.999f, 1e-8f, 0.0f});

        const double first = veda::autograd::cross_entropy(logits, {2}).value().at({});
        for (int step = 0; step < 200; ++step)
        {
            optimiser.zero_grad();
            veda::autograd::cross_entropy(logits, {2}).backward();
            optimiser.step();
        }
        const double last = veda::autograd::cross_entropy(logits, {2}).value().at({});

        CHECK(last < first);
        CHECK(last < 0.05);   // it learned to be confident about class 2
    }

    // --- a frozen parameter is skipped ------------------------------------------------------------------
    {
        Variable trainable = Variable::leaf(filled(Shape({1}), {1.0f}));
        Variable frozen = Variable::leaf(filled(Shape({1}), {1.0f}), false);
        AdamW optimiser({trainable, frozen}, AdamWConfig{0.1f, 0.9f, 0.999f, 1e-8f, 0.5f});

        set_grad(trainable, {1.0f});
        optimiser.step();

        CHECK(trainable.value().at({0}) < 1.0f);
        CHECK_EQ(frozen.value().at({0}), 1.0f);   // not even decayed — LoRA would depend on this
        optimiser.zero_grad();
    }

    // --- refusals -----------------------------------------------------------------------------------------
    {
        const Variable p = Variable::leaf(Tensor{Shape({1})});
        CHECK_THROWS_AS(AdamW({p}, AdamWConfig{0.1f, 1.0f, 0.999f, 1e-8f, 0.0f}),
                        std::invalid_argument);
        CHECK_THROWS_AS(AdamW({p}, AdamWConfig{0.1f, 0.9f, 1.5f, 1e-8f, 0.0f}),
                        std::invalid_argument);
        CHECK_THROWS_AS(AdamW({p}, AdamWConfig{0.1f, 0.9f, 0.999f, 0.0f, 0.0f}),
                        std::invalid_argument);
        CHECK_THROWS_AS(AdamW({p}, AdamWConfig{-0.1f, 0.9f, 0.999f, 1e-8f, 0.0f}),
                        std::invalid_argument);
    }

    return VEDA_TEST_SUMMARY("AdamWTest");
}
