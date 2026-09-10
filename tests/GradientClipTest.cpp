// E15.S2.T3 — clipping by global norm: shorten the step, never turn it

#include "AdamW.h"
#include "Clip.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"
#include "Variable.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using veda::autograd::Variable;
using veda::core::Shape;
using veda::core::Tensor;
using veda::optim::AdamW;
using veda::optim::AdamWConfig;
using veda::optim::clip_gradient_norm;

namespace
{
Variable parameter(const Shape& shape, const std::vector<float>& gradient,
                   bool requires_grad = true)
{
    Variable v = Variable::leaf(Tensor{shape}, requires_grad);
    for (size_t i = 0; i < gradient.size(); ++i)
    {
        v.node()->grad.data()[i] = gradient[i];
    }
    return v;
}

double global_norm(const std::vector<Variable>& parameters)
{
    double squares = 0.0;
    for (const Variable& p : parameters)
    {
        for (size_t i = 0; i < p.grad().numel(); ++i)
        {
            squares += static_cast<double>(p.grad().data()[i]) * p.grad().data()[i];
        }
    }
    return std::sqrt(squares);
}
} // namespace

int main()
{
    // --- the arithmetic example ------------------------------------------------------------------
    {
        // g_a = [3, 4], g_b = [12]  ->  norm = sqrt(9 + 16 + 144) = 13 exactly
        Variable a = parameter(Shape({2}), {3.0f, 4.0f});
        Variable b = parameter(Shape({1}), {12.0f});
        const std::vector<Variable> parameters = {a, b};

        CHECK_NEAR(global_norm(parameters), 13.0, 1e-6);

        const float reported = clip_gradient_norm(parameters, 6.5f);

        // *** the returned value is the norm BEFORE clipping — the training-log number ***
        CHECK_NEAR(reported, 13.0, 1e-5);
        CHECK_NEAR(global_norm(parameters), 6.5, 1e-5);   // exactly the threshold

        // *** and the direction survived: every ratio unchanged ***
        CHECK_NEAR(a.grad().at({0}), 1.5, 1e-5);
        CHECK_NEAR(a.grad().at({1}), 2.0, 1e-5);
        CHECK_NEAR(b.grad().at({0}), 6.0, 1e-5);
        CHECK_NEAR(a.grad().at({1}) / a.grad().at({0}), 4.0 / 3.0, 1e-5);
        CHECK_NEAR(b.grad().at({0}) / a.grad().at({0}), 12.0 / 3.0, 1e-5);
    }

    // under, and exactly at, the threshold: nothing is written
    {
        Variable a = parameter(Shape({2}), {3.0f, 4.0f});
        Variable b = parameter(Shape({1}), {12.0f});

        CHECK_NEAR(clip_gradient_norm({a, b}, 20.0f), 13.0, 1e-5);
        CHECK_EQ(a.grad().at({0}), 3.0f);   // bit for bit, not merely close
        CHECK_EQ(b.grad().at({0}), 12.0f);

        CHECK_NEAR(clip_gradient_norm({a, b}, 13.0f), 13.0, 1e-5);
        CHECK_EQ(a.grad().at({0}), 3.0f);
        CHECK_EQ(b.grad().at({0}), 12.0f);
    }

    // *** a non-finite gradient is reported, not spread across the model ***
    {
        Variable good = parameter(Shape({2}), {1.0f, 1.0f});
        Variable bad = parameter(Shape({1}), {std::numeric_limits<float>::quiet_NaN()});

        const float reported = clip_gradient_norm({good, bad}, 1.0f);
        CHECK(!std::isfinite(reported));
        CHECK_EQ(good.grad().at({0}), 1.0f);   // untouched — multiplying by NaN would poison it
        CHECK_EQ(good.grad().at({1}), 1.0f);

        Variable infinite = parameter(Shape({1}), {std::numeric_limits<float>::infinity()});
        CHECK(!std::isfinite(clip_gradient_norm({good, infinite}, 1.0f)));
        CHECK_EQ(good.grad().at({0}), 1.0f);
    }

    // frozen parameters are excluded from the norm and untouched by the scaling
    {
        Variable trainable = parameter(Shape({2}), {3.0f, 4.0f});
        Variable frozen = parameter(Shape({1}), {100.0f}, false);

        const float reported = clip_gradient_norm({trainable, frozen}, 1.0f);
        CHECK_NEAR(reported, 5.0, 1e-5);   // 100 did not enter the norm
        CHECK_NEAR(global_norm({trainable}), 1.0, 1e-5);
        CHECK_EQ(frozen.grad().at({0}), 100.0f);
    }

    // *** clipping then stepping moves less far, in the same direction ***
    {
        const AdamWConfig config{0.1f, 0.9f, 0.999f, 1e-8f, 0.0f};

        Variable clipped_a = parameter(Shape({1}), {50.0f});
        Variable clipped_b = parameter(Shape({1}), {1.0f});
        Variable free_a = parameter(Shape({1}), {50.0f});
        Variable free_b = parameter(Shape({1}), {1.0f});

        clip_gradient_norm({clipped_a, clipped_b}, 1.0f);

        AdamW clipped({clipped_a, clipped_b}, config);
        AdamW unclipped({free_a, free_b}, config);
        clipped.step();
        unclipped.step();

        // same direction: every parameter moved against its gradient in both runs
        CHECK(clipped_a.value().at({0}) < 0.0f);
        CHECK(free_a.value().at({0}) < 0.0f);
        CHECK(clipped_b.value().at({0}) < 0.0f);
        CHECK(free_b.value().at({0}) < 0.0f);

        // and the gradients that reached the optimiser were shorter, with their ratio intact
        CHECK_NEAR(clipped_a.grad().at({0}) / clipped_b.grad().at({0}), 50.0, 1e-4);
        CHECK(std::fabs(clipped_a.grad().at({0})) < std::fabs(free_a.grad().at({0})));
    }

    // a spike is bounded whatever its size — the reason the guard exists
    {
        for (const float spike : {10.0f, 1e3f, 1e6f})
        {
            Variable p = parameter(Shape({2}), {spike, spike});
            CHECK_NEAR(clip_gradient_norm({p}, 1.0f), spike * std::sqrt(2.0), spike * 1e-4);
            CHECK_NEAR(global_norm({p}), 1.0, 1e-4);
        }
    }

    // refusals
    {
        Variable p = parameter(Shape({1}), {1.0f});
        CHECK_THROWS_AS(clip_gradient_norm({p}, 0.0f), std::invalid_argument);
        CHECK_THROWS_AS(clip_gradient_norm({p}, -1.0f), std::invalid_argument);

        // no parameters at all is a norm of zero, not an error
        CHECK_EQ(clip_gradient_norm({}, 1.0f), 0.0f);
    }

    return VEDA_TEST_SUMMARY("GradientClipTest");
}
