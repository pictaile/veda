// E2.S4.T9 (the idea) and T10 (silu, gelu)

#include "Activations.h"
#include "Elementwise.h"
#include "Matmul.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <cmath>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::ops::gelu;
using veda::ops::matmul;
using veda::ops::mul;
using veda::ops::silu;

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
} // namespace

int main()
{
    // --- T9: why a non-linearity is needed at all ----------------------------------------------
    // Two stacked projections with nothing between them are one projection: W2(W1 x) == (W2 W1) x.
    // Depth only means something once something non-linear sits in between.
    {
        const Tensor x = filled(Shape({1, 2}), {1, 2});
        const Tensor w1 = filled(Shape({2, 2}), {1, 2, 3, 4});
        const Tensor w2 = filled(Shape({2, 2}), {0, 1, 1, 0});

        const Tensor stacked = matmul(matmul(x, w1), w2);
        const Tensor collapsed = matmul(x, matmul(w1, w2));
        CHECK_EQ(stacked.at({0, 0}), collapsed.at({0, 0}));
        CHECK_EQ(stacked.at({0, 1}), collapsed.at({0, 1}));

        // with silu in between, the two are no longer the same computation
        const Tensor with_activation = matmul(silu(matmul(x, w1)), w2);
        CHECK(with_activation.at({0, 0}) != collapsed.at({0, 0}));
    }

    // --- T10: silu, hand-computed --------------------------------------------------------------
    {
        const Tensor x = filled(Shape({6}), {0.0f, 1.0f, -1.0f, 2.0f, -10.0f, 100.0f});
        const Tensor y = silu(x);

        CHECK_EQ(y.shape(), Shape({6}));
        CHECK_EQ(y.at({0}), 0.0f);                  // silu(0) is exactly 0
        CHECK_NEAR(y.at({1}), 0.73106, 1e-5);
        CHECK_NEAR(y.at({2}), -0.26894, 1e-5);
        CHECK_NEAR(y.at({3}), 1.76159, 1e-5);
        CHECK_NEAR(y.at({4}), -0.00045, 1e-5);      // essentially 0
        CHECK_NEAR(y.at({5}), 100.0, 1e-4);         // essentially x

        CHECK(y.is_contiguous());
        CHECK_EQ(x.at({1}), 1.0f);                  // input untouched
    }

    // silu(x) - silu(-x) == x, since sigma(x) + sigma(-x) = 1. A free check at any magnitude.
    {
        for (const float value : {0.5f, 1.0f, 3.0f, 7.5f, 20.0f})
        {
            const Tensor pair = filled(Shape({2}), {value, -value});
            const Tensor y = silu(pair);
            CHECK_NEAR(y.at({0}) - y.at({1}), value, 1e-4);
        }
    }

    // silu is non-monotonic: it dips below zero around x = -1.28 before rising back
    {
        const Tensor sweep = filled(Shape({5}), {-3.0f, -1.5f, -1.28f, -1.0f, -0.5f});
        const Tensor y = silu(sweep);
        for (size_t i = 0; i < 5; ++i)
        {
            CHECK(y.at({i}) < 0.0f);
        }
        CHECK(y.at({2}) < y.at({0}));   // the minimum is in the middle, not at the far end
        CHECK(y.at({2}) < y.at({4}));
        CHECK_NEAR(y.at({2}), -0.2784, 1e-3);
    }

    // no NaN from exp overflow at large negative inputs — the reason the formula needs no branch
    {
        const Tensor extreme = filled(Shape({3}), {-100.0f, -700.0f, -1e30f});
        const Tensor y = silu(extreme);
        for (size_t i = 0; i < 3; ++i)
        {
            CHECK(std::isfinite(y.at({i})));
            CHECK_NEAR(y.at({i}), 0.0, 1e-6);
        }
    }

    // --- gelu ------------------------------------------------------------------------------------
    {
        const Tensor x = filled(Shape({5}), {0.0f, 1.0f, -1.0f, 2.0f, -100.0f});
        const Tensor y = gelu(x);

        CHECK_EQ(y.at({0}), 0.0f);
        CHECK_NEAR(y.at({1}), 0.84119, 1e-4);
        CHECK_NEAR(y.at({2}), -0.15881, 1e-4);
        CHECK_NEAR(y.at({3}), 1.9546, 1e-3);
        CHECK(std::isfinite(y.at({4})));
        CHECK_NEAR(y.at({4}), 0.0, 1e-6);

        // gelu and silu are close in shape but not the same function
        const Tensor same_input = filled(Shape({1}), {1.0f});
        CHECK(gelu(same_input).at({0}) != silu(same_input).at({0}));
        CHECK_NEAR(gelu(same_input).at({0}) - silu(same_input).at({0}), 0.11, 1e-2);
    }

    // --- T9: gating, as SwiGLU will use it in E9 -------------------------------------------------
    {
        const Tensor gate_scores = filled(Shape({2}), {2.0f, -2.0f});
        const Tensor up = filled(Shape({2}), {10.0f, 10.0f});

        const Tensor gated = mul(silu(gate_scores), up);
        CHECK_NEAR(gated.at({0}), 17.616, 1e-2);    // the channel passes almost fully
        CHECK_NEAR(gated.at({1}), -2.384, 1e-2);    // this one is nearly closed, and inverted

        // same up values, completely different fates — decided by the gate branch alone
        CHECK(gated.at({0}) > 7.0f * gated.at({1}) * -1.0f);
    }

    // --- shapes, views, edge cases ---------------------------------------------------------------
    {
        // shape is always preserved; activations mix nothing between positions
        const Tensor block{Shape({1, 2, 3})};
        CHECK_EQ(silu(block).shape(), Shape({1, 2, 3}));
        CHECK_EQ(gelu(block).shape(), Shape({1, 2, 3}));
        CHECK_EQ(silu(block).at({0, 1, 2}), 0.0f);

        // a non-contiguous input
        const Tensor x = filled(Shape({2, 3}), {0, 1, -1, 2, -2, 3});
        const Tensor transposed = x.transpose(0, 1);
        CHECK(!transposed.is_contiguous());
        const Tensor y = silu(transposed);
        CHECK_EQ(y.shape(), Shape({3, 2}));
        CHECK(y.is_contiguous());
        CHECK_EQ(y.at({0, 0}), 0.0f);
        CHECK_NEAR(y.at({0, 1}), 1.76159, 1e-5);    // silu(2)
        CHECK_NEAR(y.at({1, 0}), 0.73106, 1e-5);    // silu(1)

        // a rank-0 tensor and an empty one
        CHECK_EQ(silu(filled(Shape({}), {1.0f})).at({}), silu(filled(Shape({1}), {1.0f})).at({0}));
        CHECK_EQ(silu(Tensor{Shape({2, 0})}).numel(), size_t{0});
        CHECK_EQ(gelu(Tensor{Shape({0})}).shape(), Shape({0}));
    }

    return VEDA_TEST_SUMMARY("ActivationsTest");
}
