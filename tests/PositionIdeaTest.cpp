// E8.S1.T1 — why attention needs position (a Learn task; RoPE arrives in T2)

#include "Attention.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <cmath>
#include <utility>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::model::attention_scores;

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

// A 2-D rotation, written out — the operation RoPE performs on each pair.
std::pair<float, float> rotate(float x, float y, double angle)
{
    return {static_cast<float>(x * std::cos(angle) - y * std::sin(angle)),
            static_cast<float>(y * std::cos(angle) + x * std::sin(angle))};
}
} // namespace

int main()
{
    // --- attention is permutation-invariant ------------------------------------------------------
    // Permute the positions and the score matrix is permuted with them — nothing else changes,
    // because score[t,s] = q_t . k_s depends on the two vectors and on nothing else.
    {
        const Tensor q = filled(Shape({1, 1, 3, 2}), {1, 0, 0, 1, 1, 1});
        const Tensor swapped = filled(Shape({1, 1, 3, 2}), {0, 1, 1, 0, 1, 1});   // rows 0 and 1 swapped

        const Tensor a = attention_scores(q, q);
        const Tensor b = attention_scores(swapped, swapped);

        // score(0,1) in the original is score(1,0) after the swap: the same numbers, relabelled
        CHECK_EQ(a.at({0, 0, 0, 1}), b.at({0, 0, 1, 0}));
        CHECK_EQ(a.at({0, 0, 0, 2}), b.at({0, 0, 1, 2}));
        CHECK_EQ(a.at({0, 0, 2, 2}), b.at({0, 0, 2, 2}));

        // and identical tokens at different positions score identically against everything
        const Tensor repeated = filled(Shape({1, 1, 3, 2}), {2, 3, 2, 3, 1, 1});
        const Tensor scores = attention_scores(repeated, repeated);
        for (size_t s = 0; s < 3; ++s)
        {
            CHECK_EQ(scores.at({0, 0, 0, s}), scores.at({0, 0, 1, s}));
        }
    }

    // --- rotation preserves length ----------------------------------------------------------------
    {
        for (const double angle : {0.0, 0.5, 1.0, 3.14159, 10.0})
        {
            const auto [x, y] = rotate(3.0f, 4.0f, angle);
            CHECK_NEAR(std::sqrt(x * x + y * y), 5.0, 1e-5);
        }

        // position 0 rotates by nothing
        const auto [x, y] = rotate(3.0f, 4.0f, 0.0);
        CHECK_EQ(x, 3.0f);
        CHECK_EQ(y, 4.0f);
    }

    // *** the property the whole scheme rests on: the score depends only on the distance ***
    {
        const float qx = 0.6f;
        const float qy = -0.8f;
        const float kx = 0.3f;
        const float ky = 0.9f;
        const double omega = 1.0;

        auto score_at = [&](double q_position, double k_position) {
            const auto [q0, q1] = rotate(qx, qy, q_position * omega);
            const auto [k0, k1] = rotate(kx, ky, k_position * omega);
            return q0 * k0 + q1 * k1;
        };

        // same distance, same score — at completely different absolute positions
        CHECK_NEAR(score_at(5, 3), score_at(12, 10), 1e-5);
        CHECK_NEAR(score_at(1, 0), score_at(101, 100), 1e-4);
        CHECK_NEAR(score_at(0, 0), score_at(77, 77), 1e-5);

        // different distance, different score
        CHECK(std::fabs(score_at(5, 3) - score_at(5, 1)) > 0.1f);

        // and rotating BOTH by the same amount changes nothing, which is why absolute position
        // cancels out of the arithmetic
        CHECK_NEAR(score_at(0, 0), qx * kx + qy * ky, 1e-6);
    }

    // --- the frequency ladder ----------------------------------------------------------------------
    {
        const size_t head_dim = 4;
        const double theta = 10000.0;

        const double omega_0 = std::pow(theta, -0.0 / head_dim);
        const double omega_1 = std::pow(theta, -2.0 / head_dim);
        CHECK_NEAR(omega_0, 1.0, 1e-9);
        CHECK_NEAR(omega_1, 0.01, 1e-9);

        // the fast pair repeats every 2*pi positions; the slow one every 628
        CHECK_NEAR(2.0 * 3.14159265 / omega_0, 6.283, 1e-2);
        CHECK_NEAR(2.0 * 3.14159265 / omega_1, 628.3, 1e-1);

        // raising theta slows the slow end down, which is how a longer context is bought:
        // Qwen3 uses 1,000,000 where the original Llama used 10,000
        const double slow_at_10k = std::pow(10000.0, -2.0 * 63 / 128);
        const double slow_at_1m = std::pow(1000000.0, -2.0 * 63 / 128);
        CHECK(slow_at_1m < slow_at_10k);
        CHECK_NEAR(slow_at_10k / slow_at_1m, std::pow(100.0, 2.0 * 63 / 128), 1e-3);
    }

    return VEDA_TEST_SUMMARY("PositionIdeaTest");
}
