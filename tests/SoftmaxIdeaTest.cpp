// E2.S3.T7 — The softmax idea (a Learn task; ops::softmax arrives in T8)
//
// Establishes, in plain arithmetic, what softmax computes and why the maximum must be subtracted.

#include "TestSupport.h"

#include <cmath>
#include <limits>
#include <vector>

namespace
{
// The definition, written naively — exactly the version that must not ship.
std::vector<float> naive_softmax(const std::vector<float>& x)
{
    float sum = 0.0f;
    for (const float value : x)
    {
        sum += std::exp(value);
    }

    std::vector<float> out;
    for (const float value : x)
    {
        out.push_back(std::exp(value) / sum);
    }
    return out;
}

// The stable form: shift by the maximum first.
std::vector<float> stable_softmax(const std::vector<float>& x)
{
    float maximum = -std::numeric_limits<float>::infinity();
    for (const float value : x)
    {
        maximum = value > maximum ? value : maximum;
    }

    float sum = 0.0f;
    for (const float value : x)
    {
        sum += std::exp(value - maximum);
    }

    std::vector<float> out;
    for (const float value : x)
    {
        out.push_back(std::exp(value - maximum) / sum);
    }
    return out;
}

bool close(float a, float b, float tolerance = 1e-5f)
{
    return std::fabs(a - b) <= tolerance;
}

float total(const std::vector<float>& values)
{
    float sum = 0.0f;
    for (const float value : values)
    {
        sum += value;
    }
    return sum;
}
} // namespace

int main()
{
    // equal inputs give a uniform distribution — the one case checkable without a calculator
    {
        const std::vector<float> uniform = stable_softmax({1.0f, 1.0f, 1.0f});
        CHECK(close(uniform[0], 1.0f / 3.0f));
        CHECK(close(uniform[1], 1.0f / 3.0f));
        CHECK(close(uniform[2], 1.0f / 3.0f));
        CHECK(close(total(uniform), 1.0f));

        // only differences matter, so [5,5,5] gives the same answer
        const std::vector<float> same = stable_softmax({5.0f, 5.0f, 5.0f});
        CHECK(close(same[0], 1.0f / 3.0f));
    }

    // the worked example
    {
        const std::vector<float> p = stable_softmax({2.0f, 1.0f, 0.1f});
        CHECK(close(p[0], 0.659f, 1e-3f));
        CHECK(close(p[1], 0.242f, 1e-3f));
        CHECK(close(p[2], 0.099f, 1e-3f));
        CHECK(close(total(p), 1.0f));

        // every output is a probability
        for (const float value : p)
        {
            CHECK(value > 0.0f && value < 1.0f);
        }
        // and the order is preserved: the largest score keeps the largest share
        CHECK(p[0] > p[1] && p[1] > p[2]);
    }

    // shift invariance — the property that licenses subtracting the maximum
    {
        const std::vector<float> base = stable_softmax({1.0f, 2.0f, 3.0f});
        for (const float shift : {-100.0f, -1.0f, 0.0f, 50.0f})
        {
            const std::vector<float> shifted =
                stable_softmax({1.0f + shift, 2.0f + shift, 3.0f + shift});
            CHECK(close(shifted[0], base[0]));
            CHECK(close(shifted[2], base[2]));
        }

        // ratios are NOT preserved: [1,2] and [10,20] are different distributions
        CHECK(!close(stable_softmax({1.0f, 2.0f})[0], stable_softmax({10.0f, 20.0f})[0]));
    }

    // where exp breaks in fp32, and what the naive version does about it
    {
        CHECK(std::isfinite(std::exp(88.0f)));
        CHECK(std::isinf(std::exp(89.0f)));      // fp32 overflows just past 88
        CHECK(std::isinf(std::exp(1000.0f)));

        const std::vector<float> poisoned = naive_softmax({1000.0f, 1001.0f});
        CHECK(std::isnan(poisoned[0]));          // inf / inf
        CHECK(std::isnan(poisoned[1]));

        // the stable form returns the same distribution as the small-numbered case
        const std::vector<float> rescued = stable_softmax({1000.0f, 1001.0f});
        CHECK(!std::isnan(rescued[0]));
        CHECK(close(rescued[0], 0.269f, 1e-3f));
        CHECK(close(rescued[1], 0.731f, 1e-3f));
        CHECK(close(rescued[0], stable_softmax({1.0f, 2.0f})[0]));
        CHECK(close(total(rescued), 1.0f));
    }

    // in the stable form the largest argument is exactly 0, so exp(0) = 1 and the denominator is
    // never smaller than 1 — no overflow above, no division by zero below
    {
        CHECK_EQ(std::exp(0.0f), 1.0f);
        const std::vector<float> extreme = stable_softmax({-1000.0f, 1000.0f});
        CHECK(close(extreme[1], 1.0f));
        CHECK(close(extreme[0], 0.0f));     // underflow to zero is harmless and correct
        CHECK(!std::isnan(extreme[0]));
    }

    // a masked position: exp(-inf - m) is exactly 0, which is what E7 masks with
    {
        const float minus_infinity = -std::numeric_limits<float>::infinity();
        const std::vector<float> masked = stable_softmax({1.0f, minus_infinity, 2.0f});
        CHECK_EQ(masked[1], 0.0f);          // exactly zero, not merely small
        CHECK(close(total(masked), 1.0f));
        CHECK(close(masked[0], stable_softmax({1.0f, 2.0f})[0]));   // as if it were not there
    }

    return VEDA_TEST_SUMMARY("SoftmaxIdeaTest");
}
