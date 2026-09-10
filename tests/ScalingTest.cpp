// E7.S3.T4 — scaling the scores by 1/sqrt(d_k)

#include "Attention.h"
#include "Elementwise.h"
#include "Shape.h"
#include "Softmax.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::model::attention_scores;
using veda::model::scale_scores;
using veda::ops::softmax;

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

// Components with mean 0 and variance 1 — roughly what normalised activations look like.
Tensor normal_tensor(const Shape& shape, std::mt19937& engine)
{
    std::normal_distribution<float> values(0.0f, 1.0f);
    Tensor t{shape};
    for (size_t i = 0; i < shape.size(); ++i)
    {
        t.data()[i] = values(engine);
    }
    return t;
}

double standard_deviation(const Tensor& t)
{
    double sum = 0.0;
    for (size_t i = 0; i < t.numel(); ++i)
    {
        sum += t.data()[i];
    }
    const double mean = sum / static_cast<double>(t.numel());

    double squares = 0.0;
    for (size_t i = 0; i < t.numel(); ++i)
    {
        const double difference = t.data()[i] - mean;
        squares += difference * difference;
    }
    return std::sqrt(squares / static_cast<double>(t.numel()));
}
} // namespace

int main()
{
    // the hand-computed case: sqrt(4) = 2
    {
        const Tensor scores = filled(Shape({2, 2}), {4, 8, 12, 16});
        const Tensor scaled = scale_scores(scores, 4);

        CHECK_EQ(scaled.shape(), Shape({2, 2}));
        CHECK_EQ(scaled.at({0, 0}), 2.0f);
        CHECK_EQ(scaled.at({0, 1}), 4.0f);
        CHECK_EQ(scaled.at({1, 0}), 6.0f);
        CHECK_EQ(scaled.at({1, 1}), 8.0f);

        CHECK(scaled.is_contiguous());
        CHECK_EQ(scores.at({0, 0}), 4.0f);              // input unmodified
        CHECK(scaled.storage() != scores.storage());

        // head_dim 1 changes nothing; 0 is refused
        CHECK_EQ(scale_scores(scores, 1).at({1, 1}), 16.0f);
        CHECK_THROWS_AS(scale_scores(scores, 0), std::invalid_argument);

        // shape-preserving at the rank the model uses
        CHECK_EQ(scale_scores(Tensor{Shape({1, 16, 8, 8})}, 128).shape(), Shape({1, 16, 8, 8}));
        CHECK_EQ(scale_scores(Tensor{Shape({0, 3})}, 4).numel(), size_t{0});

        // a head_dim that is not a perfect square
        CHECK_NEAR(scale_scores(filled(Shape({1}), {10.0f}), 5).at({0}), 10.0 / std::sqrt(5.0), 1e-5);
    }

    // *** the property this task exists for: score variance against the head dimension ***
    {
        std::mt19937 engine(20260901);
        const size_t T = 64;

        for (const size_t head_dim : {16u, 64u, 256u})
        {
            const Tensor q = normal_tensor(Shape({1, 1, T, head_dim}), engine);
            const Tensor k = normal_tensor(Shape({1, 1, T, head_dim}), engine);
            const Tensor raw = attention_scores(q, k);

            // unscaled: the standard deviation grows as sqrt(Dh), because the variances of Dh
            // independent products add
            const double raw_std = standard_deviation(raw);
            const double expected = std::sqrt(static_cast<double>(head_dim));
            CHECK(raw_std > expected * 0.85 && raw_std < expected * 1.15);

            // scaled: the same distribution at every head dimension
            const double scaled_std = standard_deviation(scale_scores(raw, head_dim));
            CHECK(scaled_std > 0.85 && scaled_std < 1.15);

            // the wrong constant, 1/d_k: it shrinks to 1/sqrt(Dh) and fails the other way
            const double over_scaled = standard_deviation(veda::ops::mul(
                raw, 1.0f / static_cast<float>(head_dim)));
            CHECK(over_scaled < 1.0 / (expected * 0.85));
            CHECK(over_scaled > 1.0 / (expected * 1.15));
        }
    }

    // what unscaled scores do to softmax, and what scaling restores
    {
        std::mt19937 engine(31415);
        const size_t head_dim = 256;
        const size_t T = 32;

        const Tensor q = normal_tensor(Shape({1, 1, T, head_dim}), engine);
        const Tensor k = normal_tensor(Shape({1, 1, T, head_dim}), engine);
        const Tensor raw = attention_scores(q, k);

        const Tensor saturated = softmax(raw, 3);
        const Tensor spread = softmax(scale_scores(raw, head_dim), 3);

        // How concentrated each row is, averaged over all of them: the mean of each row's largest
        // weight. Uniform attention over 32 positions would give 1/32 = 0.031.
        double saturated_peak = 0.0;
        double spread_peak = 0.0;
        for (size_t t = 0; t < T; ++t)
        {
            float saturated_row = 0.0f;
            float spread_row = 0.0f;
            for (size_t s = 0; s < T; ++s)
            {
                saturated_row = std::max(saturated_row, saturated.at({0, 0, t, s}));
                spread_row = std::max(spread_row, spread.at({0, 0, t, s}));
            }
            saturated_peak += saturated_row;
            spread_peak += spread_row;
        }
        saturated_peak /= static_cast<double>(T);
        spread_peak /= static_cast<double>(T);

        // Measured here: about 0.88 unscaled against 0.15 scaled. The thresholds are loose because
        // the exact numbers depend on the draw; the order of magnitude is the point.
        CHECK(saturated_peak > 0.6);            // unscaled: most of the row's mass on one position
        CHECK(spread_peak < 0.3);               // scaled: the whole sequence still contributes
        CHECK(saturated_peak > 3.0 * spread_peak);

        // both are still distributions
        float saturated_sum = 0.0f;
        float spread_sum = 0.0f;
        for (size_t s = 0; s < T; ++s)
        {
            saturated_sum += saturated.at({0, 0, 0, s});
            spread_sum += spread.at({0, 0, 0, s});
        }
        CHECK_NEAR(saturated_sum, 1.0, 1e-5);
        CHECK_NEAR(spread_sum, 1.0, 1e-5);

        // and scaling does not change WHICH position is preferred, only by how much
        size_t best_raw = 0;
        size_t best_scaled = 0;
        for (size_t s = 1; s < T; ++s)
        {
            best_raw = saturated.at({0, 0, 0, s}) > saturated.at({0, 0, 0, best_raw}) ? s : best_raw;
            best_scaled = spread.at({0, 0, 0, s}) > spread.at({0, 0, 0, best_scaled}) ? s : best_scaled;
        }
        CHECK_EQ(best_raw, best_scaled);
    }

    // the worked softmax example from the task
    {
        const Tensor row = filled(Shape({3}), {8.0f, 6.0f, 7.0f});
        const Tensor raw_weights = softmax(row, 0);
        CHECK_NEAR(raw_weights.at({0}), 0.665, 1e-3);
        CHECK_NEAR(raw_weights.at({1}), 0.090, 1e-3);

        // scaled by 1/8: [1.0, 0.75, 0.875]
        const Tensor scaled_weights = softmax(scale_scores(row, 64), 0);
        CHECK_NEAR(scaled_weights.at({0}), 0.3758, 1e-3);
        CHECK_NEAR(scaled_weights.at({1}), 0.2926, 1e-3);
        CHECK_NEAR(scaled_weights.at({2}), 0.3316, 1e-3);
        CHECK(scaled_weights.at({0}) > scaled_weights.at({2}));   // still prefers the same one
    }

    // head_dim, not hidden_size: the difference is a silent factor of sqrt(H)
    {
        const Tensor scores = filled(Shape({1}), {16.0f});
        const size_t H = 16;
        const size_t Dh = 128;
        const size_t D = H * Dh;

        const float correct = scale_scores(scores, Dh).at({0});
        const float wrong = scale_scores(scores, D).at({0});
        CHECK_NEAR(correct / wrong, std::sqrt(static_cast<double>(H)), 1e-4);
        CHECK_NEAR(correct / wrong, 4.0, 1e-4);
    }

    return VEDA_TEST_SUMMARY("ScalingTest");
}
