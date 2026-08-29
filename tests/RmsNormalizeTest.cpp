// E2.S4.T11 (the idea) and T12 (rms_normalize)

#include "Normalize.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <cmath>
#include <stdexcept>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::ops::rms_normalize;

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

// LayerNorm, written here only to make the contrast with RMSNorm concrete. Veda does not
// implement it: Qwen3 does not use it.
std::vector<float> layer_norm(const std::vector<float>& x, float eps)
{
    float mean = 0.0f;
    for (const float value : x)
    {
        mean += value;
    }
    mean /= static_cast<float>(x.size());

    float variance = 0.0f;
    for (const float value : x)
    {
        variance += (value - mean) * (value - mean);
    }
    variance /= static_cast<float>(x.size());

    std::vector<float> out;
    for (const float value : x)
    {
        out.push_back((value - mean) / std::sqrt(variance + eps));
    }
    return out;
}

float mean_square(const Tensor& t)
{
    float sum = 0.0f;
    for (size_t i = 0; i < t.numel(); ++i)
    {
        sum += t.data()[i] * t.data()[i];
    }
    return sum / static_cast<float>(t.numel());
}
} // namespace

int main()
{
    // --- T12: the hand-computed case ------------------------------------------------------------
    // x = [3, 4]:  mean square = 12.5,  rms = 3.5355,  out = [0.8485, 1.1314]
    {
        const Tensor x = filled(Shape({2}), {3.0f, 4.0f});
        const Tensor y = rms_normalize(x, 0.0f);

        CHECK_EQ(y.shape(), Shape({2}));
        CHECK_NEAR(y.at({0}), 0.84853, 1e-5);
        CHECK_NEAR(y.at({1}), 1.13137, 1e-5);
        CHECK_NEAR(mean_square(y), 1.0, 1e-5);      // the invariant: RMS of the output is 1
        CHECK(y.is_contiguous());
        CHECK_EQ(x.at({0}), 3.0f);                  // input untouched
    }

    // a constant vector normalises to exactly ones — magnitude discarded, direction kept
    {
        const Tensor ones = rms_normalize(filled(Shape({3}), {5, 5, 5}), 0.0f);
        CHECK_NEAR(ones.at({0}), 1.0, 1e-6);
        CHECK_NEAR(ones.at({1}), 1.0, 1e-6);
        CHECK_NEAR(ones.at({2}), 1.0, 1e-6);

        const Tensor negative = rms_normalize(filled(Shape({2}), {-2, -2}), 0.0f);
        CHECK_NEAR(negative.at({0}), -1.0, 1e-6);   // sign survives
        CHECK_NEAR(negative.at({1}), -1.0, 1e-6);
    }

    // --- T11: the contrast with LayerNorm --------------------------------------------------------
    // Same input, entirely different answers — the difference is not subtle
    {
        const std::vector<float> constant = {5.0f, 5.0f, 5.0f};
        // eps must be non-zero here: the variance of a constant vector is 0, and LayerNorm
        // without eps divides by it — the same protection RMSNorm's eps provides
        const std::vector<float> centred = layer_norm(constant, 1e-6f);
        const Tensor scaled = rms_normalize(filled(Shape({3}), constant), 0.0f);

        CHECK_NEAR(centred[0], 0.0, 1e-6);          // LayerNorm: the mean is subtracted, nothing left
        CHECK_NEAR(scaled.at({0}), 1.0, 1e-6);      // RMSNorm: all ones

        // on a non-constant vector they differ too, because RMSNorm keeps the offset
        const std::vector<float> x = {1.0f, 2.0f, 3.0f};
        const std::vector<float> ln = layer_norm(x, 1e-6f);
        const Tensor rn = rms_normalize(filled(Shape({3}), x), 0.0f);
        CHECK(ln[0] < 0.0f);                        // centring makes the smallest entry negative
        CHECK(rn.at({0}) > 0.0f);                   // RMSNorm never changes a sign
    }

    // scale invariance: the point of the operation
    {
        const Tensor x = filled(Shape({4}), {1, -2, 3, -4});
        const Tensor base = rms_normalize(x, 0.0f);

        for (const float factor : {0.001f, 2.0f, 100.0f, 1e6f})
        {
            std::vector<float> scaled_values;
            for (const float value : {1.0f, -2.0f, 3.0f, -4.0f})
            {
                scaled_values.push_back(value * factor);
            }
            const Tensor scaled = rms_normalize(filled(Shape({4}), scaled_values), 0.0f);
            for (size_t i = 0; i < 4; ++i)
            {
                CHECK_NEAR(scaled.at({i}), base.at({i}), 1e-4);
            }
        }
    }

    // per-token: each lane along the last axis normalises independently
    {
        // row 1 is twice row 0, so both normalise to the same thing
        const Tensor x = filled(Shape({2, 2}), {3, 4, 6, 8});
        const Tensor y = rms_normalize(x, 0.0f);

        CHECK_EQ(y.shape(), Shape({2, 2}));
        CHECK_NEAR(y.at({0, 0}), 0.84853, 1e-5);
        CHECK_NEAR(y.at({0, 1}), 1.13137, 1e-5);
        CHECK_NEAR(y.at({1, 0}), 0.84853, 1e-5);
        CHECK_NEAR(y.at({1, 1}), 1.13137, 1e-5);

        // the [B, T, D] shape the model uses: B*T lanes of length D
        Tensor activations{Shape({1, 3, 4})};
        for (size_t i = 0; i < activations.numel(); ++i)
        {
            activations.data()[i] = static_cast<float>(i + 1);
        }
        const Tensor normalised = rms_normalize(activations, 1e-6f);
        CHECK_EQ(normalised.shape(), Shape({1, 3, 4}));
        for (size_t t = 0; t < 3; ++t)
        {
            float sum_of_squares = 0.0f;
            for (size_t d = 0; d < 4; ++d)
            {
                const float value = normalised.at({0, t, d});
                sum_of_squares += value * value;
            }
            CHECK_NEAR(sum_of_squares / 4.0, 1.0, 1e-4);   // every token lane has RMS 1
        }
    }

    // eps: what it protects against
    {
        // an all-zero lane would divide by zero without it
        const Tensor zeros = rms_normalize(filled(Shape({3}), {0, 0, 0}), 1e-6f);
        CHECK(!std::isnan(zeros.at({0})));
        CHECK_EQ(zeros.at({0}), 0.0f);
        CHECK_EQ(zeros.at({2}), 0.0f);

        // a nearly-zero lane is amplified, but boundedly
        const Tensor tiny = rms_normalize(filled(Shape({2}), {1e-9f, 1e-9f}), 1e-6f);
        CHECK(std::isfinite(tiny.at({0})));
        CHECK(tiny.at({0}) < 1.0f);

        // with eps = 0 the same input reaches 1.0, which is what eps is bounding
        const Tensor unbounded = rms_normalize(filled(Shape({2}), {1e-9f, 1e-9f}), 0.0f);
        CHECK_NEAR(unbounded.at({0}), 1.0, 1e-4);
    }

    // non-contiguous input
    {
        const Tensor x = filled(Shape({2, 2}), {3, 6, 4, 8});
        const Tensor transposed = x.transpose(0, 1);   // rows become [3,4] and [6,8]
        CHECK(!transposed.is_contiguous());

        const Tensor y = rms_normalize(transposed, 0.0f);
        CHECK(y.is_contiguous());
        CHECK_NEAR(y.at({0, 0}), 0.84853, 1e-5);
        CHECK_NEAR(y.at({1, 1}), 1.13137, 1e-5);

        // a sliced input, with a non-zero offset
        const Tensor wide = filled(Shape({1, 4}), {9, 3, 4, 9});
        const Tensor window = wide.slice(1, 1, 2);     // [3, 4], offset 1
        const Tensor sliced = rms_normalize(window, 0.0f);
        CHECK_NEAR(sliced.at({0, 0}), 0.84853, 1e-5);
        CHECK_NEAR(sliced.at({0, 1}), 1.13137, 1e-5);
    }

    // edge cases
    {
        // a lane of length 1 normalises to +-1
        const Tensor single = rms_normalize(filled(Shape({2, 1}), {7, -3}), 0.0f);
        CHECK_NEAR(single.at({0, 0}), 1.0, 1e-5);
        CHECK_NEAR(single.at({1, 0}), -1.0, 1e-5);

        // a zero-length last dimension leaves an empty result
        CHECK_EQ(rms_normalize(Tensor{Shape({2, 0})}, 1e-6f).shape(), Shape({2, 0}));
        CHECK_EQ(rms_normalize(Tensor{Shape({2, 0})}, 1e-6f).numel(), size_t{0});

        // a rank-0 tensor has no last dimension
        CHECK_THROWS_AS(rms_normalize(Tensor{Shape({})}, 1e-6f), std::invalid_argument);

        // large magnitudes stay finite
        const Tensor large = rms_normalize(filled(Shape({2}), {1e18f, 2e18f}), 0.0f);
        CHECK(std::isfinite(large.at({0})));
        CHECK_NEAR(mean_square(large), 1.0, 1e-4);
    }

    return VEDA_TEST_SUMMARY("RmsNormalizeTest");
}
