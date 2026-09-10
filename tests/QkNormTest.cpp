// E8.S3.T3 — QK-Norm, and the pipeline order

#include "Attention.h"
#include "Compare.h"
#include "Linear.h"
#include "Normalize.h"
#include "RMSNorm.h"
#include "Rope.h"
#include "ScaledDotProductAttention.h"
#include "Shape.h"
#include "Softmax.h"
#include "TensorFile.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::model::apply_rope;
using veda::model::rope_tables;
using veda::model::ScaledDotProductAttention;
using veda::nn::Linear;
using veda::nn::RMSNorm;

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

Tensor identity(size_t n)
{
    Tensor t{Shape({n, n})};
    for (size_t i = 0; i < n; ++i)
    {
        t.data()[i * n + i] = 1.0f;
    }
    return t;
}

Tensor ones(size_t n)
{
    Tensor t{Shape({n})};
    for (size_t i = 0; i < n; ++i)
    {
        t.data()[i] = 1.0f;
    }
    return t;
}

const float eps = 1e-6f;
} // namespace

int main()
{
    // --- the hand-computed example: q = [3,4,0,0], gamma = 1, position 1 --------------------------
    {
        const Tensor q = filled(Shape({1, 1, 2, 4}), {3, 4, 0, 0, 3, 4, 0, 0});
        const RMSNorm norm(ones(4), eps);

        const Tensor normalised = norm.forward(q);
        CHECK_NEAR(normalised.at({0, 0, 0, 0}), 1.2, 1e-4);   // 3 / 2.5
        CHECK_NEAR(normalised.at({0, 0, 0, 1}), 1.6, 1e-4);   // 4 / 2.5

        const Tensor rotated = apply_rope(normalised, rope_tables(2, 4, 10000.0f));
        CHECK_NEAR(rotated.at({0, 0, 1, 0}), 0.648, 1e-3);
        CHECK_NEAR(rotated.at({0, 0, 1, 1}), 1.600, 1e-3);
        CHECK_NEAR(rotated.at({0, 0, 1, 2}), 1.010, 1e-3);
        CHECK_NEAR(rotated.at({0, 0, 1, 3}), 0.016, 1e-3);
    }

    // *** the trap: with gamma = 1 the two orders agree, so a test at gamma = 1 proves nothing ***
    {
        const Tensor q = filled(Shape({1, 1, 2, 4}), {3, 4, 1, 2, -1, 5, 2, 0});
        const RMSNorm unit(ones(4), eps);
        const auto tables = rope_tables(2, 4, 10000.0f);

        const Tensor norm_then_rotate = apply_rope(unit.forward(q), tables);
        const Tensor rotate_then_norm = unit.forward(apply_rope(q, tables));

        for (size_t p = 0; p < 2; ++p)
        {
            for (size_t d = 0; d < 4; ++d)
            {
                CHECK_NEAR(norm_then_rotate.at({0, 0, p, d}), rotate_then_norm.at({0, 0, p, d}),
                           1e-5);
            }
        }
    }

    // *** and with gamma != 1 they diverge — because rotation mixes channels within a pair and
    //     gamma scales those channels differently ***
    {
        const Tensor q = filled(Shape({1, 1, 2, 4}), {3, 4, 1, 2, -1, 5, 2, 0});
        const RMSNorm scaled(filled(Shape({4}), {2.0f, 1.0f, 1.0f, 1.0f}), eps);
        const auto tables = rope_tables(2, 4, 10000.0f);

        const Tensor norm_then_rotate = apply_rope(scaled.forward(q), tables);
        const Tensor rotate_then_norm = scaled.forward(apply_rope(q, tables));

        bool differs = false;
        for (size_t d = 0; d < 4; ++d)
        {
            if (std::fabs(norm_then_rotate.at({0, 0, 1, d}) - rotate_then_norm.at({0, 0, 1, d})) >
                1e-3f)
            {
                differs = true;
            }
        }
        CHECK(differs);

        // Even the lengths differ once gamma is not 1 — because scaling per channel does not
        // commute with a rotation that mixes channels. With gamma = 1 both the values and the
        // lengths agree, which is exactly what makes the wrong order easy to ship.
        double a = 0.0;
        double b = 0.0;
        for (size_t d = 0; d < 4; ++d)
        {
            a += static_cast<double>(norm_then_rotate.at({0, 0, 1, d})) *
                 norm_then_rotate.at({0, 0, 1, d});
            b += static_cast<double>(rotate_then_norm.at({0, 0, 1, d})) *
                 rotate_then_norm.at({0, 0, 1, d});
        }
        CHECK(std::fabs(std::sqrt(a) - std::sqrt(b)) > 1e-3);
    }

    // *** the layer does norm-then-rope, pinned with a gamma that is not all ones ***
    {
        const size_t D = 4;
        const size_t DH = 4;
        const Tensor gamma = filled(Shape({DH}), {2.0f, 1.0f, 0.5f, 1.0f});

        ScaledDotProductAttention attention(
            {Linear(identity(D)), Linear(identity(D)), Linear(identity(D)), Linear(identity(D)),
             RMSNorm(gamma, eps), RMSNorm(gamma, eps)},
            {1, 1, DH, 10000.0f});

        const Tensor x = filled(Shape({1, 2, D}), {3, 4, 1, 2, -1, 5, 2, 0});
        const Tensor out = attention.forward(x);
        CHECK_EQ(out.shape(), Shape({1, 2, D}));

        // reproduce the pipeline by hand, in the layer's order
        const RMSNorm norm(gamma, eps);
        const auto tables = rope_tables(2, DH, 10000.0f);
        const Tensor q = apply_rope(norm.forward(veda::model::split_heads(x, 1, DH)), tables);
        const Tensor k = q;
        const Tensor v = veda::model::split_heads(x, 1, DH);   // V is neither normed nor rotated

        const Tensor weights = veda::ops::softmax(
            veda::model::apply_causal_mask(
                veda::model::scale_scores(veda::model::attention_scores(q, k), DH)),
            3);
        const Tensor expected =
            veda::model::merge_heads(veda::model::attention_context(weights, v));

        for (size_t t = 0; t < 2; ++t)
        {
            for (size_t d = 0; d < D; ++d)
            {
                CHECK_NEAR(out.at({0, t, d}), expected.at({0, t, d}), 1e-5);
            }
        }

        // and the OTHER order gives a different answer — so the test above is not vacuous
        const Tensor wrong_q =
            norm.forward(apply_rope(veda::model::split_heads(x, 1, DH), tables));
        const Tensor wrong_weights = veda::ops::softmax(
            veda::model::apply_causal_mask(
                veda::model::scale_scores(veda::model::attention_scores(wrong_q, wrong_q), DH)),
            3);
        const Tensor wrong =
            veda::model::merge_heads(veda::model::attention_context(wrong_weights, v));

        bool orders_differ = false;
        for (size_t d = 0; d < D; ++d)
        {
            if (std::fabs(expected.at({0, 1, d}) - wrong.at({0, 1, d})) > 1e-4f)
            {
                orders_differ = true;
            }
        }
        CHECK(orders_differ);
    }

    // QK-Norm acts per head over Dh, not across the whole [H*Dh] row
    {
        const size_t D = 4;
        const size_t H = 2;
        const size_t DH = 2;

        ScaledDotProductAttention attention(
            {Linear(identity(D)), Linear(identity(D)), Linear(identity(D)), Linear(identity(D)),
             RMSNorm(ones(DH), eps), RMSNorm(ones(DH), eps)},
            {H, H, DH, std::nullopt});

        // head 0 sees [3,4] and head 1 sees [30,40]: normalised per head they become identical
        const Tensor x = filled(Shape({1, 1, D}), {3, 4, 30, 40});
        const Tensor out = attention.forward(x);
        CHECK_EQ(out.shape(), Shape({1, 1, D}));

        // with T = 1 the only weight is 1, so the output is V — unnormalised, so the heads differ
        CHECK_NEAR(out.at({0, 0, 0}), 3.0, 1e-4);
        CHECK_NEAR(out.at({0, 0, 2}), 30.0, 1e-4);

        // a gamma of the wrong length is refused
        CHECK_THROWS_AS(ScaledDotProductAttention(
                            {Linear(identity(D)), Linear(identity(D)), Linear(identity(D)),
                             Linear(identity(D)), RMSNorm(ones(D), eps), std::nullopt},
                            {H, H, DH, std::nullopt}),
                        std::invalid_argument);
    }

    // V is neither normalised nor rotated: a layer with both produces the same V as one without
    {
        const size_t D = 4;
        const Tensor x = filled(Shape({1, 3, D}), {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3});
        const Tensor gamma = filled(Shape({D}), {3.0f, 0.2f, 1.0f, 5.0f});

        // T = 1 isolates V: the single weight is 1, so out == V exactly
        const Tensor single = filled(Shape({1, 1, D}), {1, 2, 3, 4});

        ScaledDotProductAttention plain(
            {Linear(identity(D)), Linear(identity(D)), Linear(identity(D)), Linear(identity(D))},
            {1, 1, D, std::nullopt});
        ScaledDotProductAttention decorated(
            {Linear(identity(D)), Linear(identity(D)), Linear(identity(D)), Linear(identity(D)),
             RMSNorm(gamma, eps), RMSNorm(gamma, eps)},
            {1, 1, D, 10000.0f});

        for (size_t d = 0; d < D; ++d)
        {
            CHECK_NEAR(plain.forward(single).at({0, 0, d}), decorated.forward(single).at({0, 0, d}),
                       1e-5);
            CHECK_NEAR(decorated.forward(single).at({0, 0, d}), single.at({0, 0, d}), 1e-5);
        }
    }

    // *** the position offset, and what it does NOT do ***
    // Shifting a whole chunk to a later absolute position changes nothing: every pair of positions
    // keeps its distance, and RoPE scores depend only on distance (E8.S1.T1). That is the property
    // the encoding exists for, and here it is at the layer level.
    {
        const size_t D = 4;
        ScaledDotProductAttention attention(
            {Linear(identity(D)), Linear(identity(D)), Linear(identity(D)), Linear(identity(D))},
            {1, 1, D, 10000.0f});

        const Tensor two = filled(Shape({1, 2, D}), {1, 0, 0, 0, 0, 1, 0, 0});
        const Tensor at_zero = attention.forward(two, 0);
        const Tensor at_five = attention.forward(two, 5);

        for (size_t t = 0; t < 2; ++t)
        {
            for (size_t d = 0; d < D; ++d)
            {
                CHECK_NEAR(at_zero.at({0, t, d}), at_five.at({0, t, d}), 1e-4);
            }
        }

        // The offset matters when queries and keys sit at DIFFERENT offsets — one new query against
        // a cached history, which is E12's shape. Here, directly on the rotation:
        // Both must have a non-zero pair 0 (channels 0 and 2 under the half-split pairing), or
        // the dot product is identically zero and the test measures nothing.
        const Tensor q = filled(Shape({1, 1, 1, D}), {1, 0, 1, 0});
        const Tensor k = filled(Shape({1, 1, 1, D}), {1, 1, 0, 0});

        auto score = [&](size_t q_position, size_t k_position) {
            const Tensor rq = apply_rope(q, rope_tables(1, D, 10000.0f, q_position));
            const Tensor rk = apply_rope(k, rope_tables(1, D, 10000.0f, k_position));
            double sum = 0.0;
            for (size_t d = 0; d < D; ++d)
            {
                sum += static_cast<double>(rq.at({0, 0, 0, d})) * rk.at({0, 0, 0, d});
            }
            return sum;
        };

        CHECK_NEAR(score(5, 3), score(9, 7), 1e-4);          // same distance
        CHECK(std::fabs(score(5, 3) - score(5, 4)) > 1e-3);  // different distance

        // and a chunk at offset 0 leaves position 0 unrotated
        const Tensor one = filled(Shape({1, 1, D}), {1, 0, 0, 0});
        CHECK_NEAR(attention.forward(one, 0).at({0, 0, 0}), 1.0, 1e-5);
    }

    // without norms and without rope, the layer is exactly what E7 built
    {
        const size_t D = 2;
        ScaledDotProductAttention e7_style(
            {Linear(identity(D)), Linear(identity(D)),
             Linear(filled(Shape({2, 2}), {10, 0, 0, 10})), Linear(identity(D))},
            {1, 1, 2, std::nullopt});

        const Tensor x = filled(Shape({1, 3, D}), {1, 0, 0, 1, 1, 1});
        const Tensor out = e7_style.forward(x);
        CHECK_NEAR(out.at({0, 0, 0}), 10.0, 1e-4);     // the milestone numbers from E7
        CHECK_NEAR(out.at({0, 1, 0}), 3.302, 1e-3);
        CHECK_NEAR(out.at({0, 2, 0}), 7.517, 1e-3);
    }

    // --- the reference comparison, which is the only thing that can confirm the ORDER -------------
    // A missing reference directory means "not run", never "passed" (E3.S2.T4).
    {
        const std::filesystem::path reference = "reference/layer_00_attn_out.bin";
        if (std::filesystem::exists(reference))
        {
            const Tensor expected = veda::testing::read_tensor(reference.string());
            std::cout << "  reference found: " << expected.shape().to_string() << "\n";
            CHECK(expected.rank() == 3);
            // The real comparison needs the model's weights, which arrive in E10 — this is the
            // hook, and it is exercised there.
        }
        else
        {
            std::cout << "  SKIPPED: no reference/ directory — run tools/dump_reference.py first.\n"
                      << "  The QK-Norm/RoPE order cannot be confirmed by hand; only this can.\n";
        }
    }

    return VEDA_TEST_SUMMARY("QkNormTest");
}
