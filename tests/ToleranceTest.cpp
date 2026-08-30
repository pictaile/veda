// E3.S1.T1 — Absolute and relative tolerance (a Learn task; assert_close arrives in T2)

#include "Matmul.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <cmath>
#include <limits>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::ops::matmul;

namespace
{
// The combined rule everyone uses: numpy, torch, and T2.
bool within(float actual, float expected, float rtol, float atol)
{
    return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

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
    // fp32 addition is not associative: the same numbers, a different order, a different answer
    {
        // (a + b) + c is not a + (b + c): in the second grouping the 1.0 is rounded away
        const float a = 1e8f;
        const float b = -1e8f;
        const float c = 1.0f;
        CHECK_EQ((a + b) + c, 1.0f);
        CHECK_EQ(a + (b + c), 0.0f);            // b + c rounds straight back to -1e8
        CHECK((a + b) + c != a + (b + c));

        // and 1.0 simply does not fit next to 1e8 in 24 bits of mantissa
        CHECK_EQ((a + 1.0f) - a, 0.0f);

        // which is why a sum's result depends on the order it is accumulated in
        const std::vector<float> values = {1e8f, 1.0f, -1e8f, 1.0f};
        float forward = 0.0f;
        for (const float value : values)
        {
            forward += value;
        }
        float backward = 0.0f;
        for (size_t i = values.size(); i-- > 0;)
        {
            backward += values[i];
        }
        CHECK(forward != backward);   // 1.0 versus 2.0, from the same four numbers
    }

    // machine epsilon: the relative error one fp32 operation can contribute
    {
        const float epsilon = std::numeric_limits<float>::epsilon();
        CHECK_NEAR(epsilon, 1.19e-7, 1e-9);
        CHECK_EQ(1.0f + epsilon / 2.0f, 1.0f);       // below epsilon, addition changes nothing
        CHECK(1.0f + epsilon != 1.0f);
    }

    // the three cases from the task, decided by the rule
    {
        // rounding noise at magnitude 1 — must pass
        CHECK(within(1.0000001f, 1.0f, 1e-5f, 1e-8f));

        // two numbers that are both zero in context: 100% relative, irrelevant absolute
        CHECK(!within(2e-9f, 1e-9f, 1e-5f, 0.0f));   // relative alone rejects it
        CHECK(within(2e-9f, 1e-9f, 1e-5f, 1e-8f));   // atol is what saves it

        // a real difference at magnitude 12.5 — must fail
        CHECK(!within(12.6f, 12.5f, 1e-5f, 1e-8f));

        // and what atol alone cannot do: the same relative error at a large magnitude
        CHECK(within(1.0000001f, 1.0f, 1e-5f, 0.0f));
        CHECK(within(1000000.1f, 1000000.0f, 1e-5f, 0.0f));   // rtol scales, atol would not
        CHECK(!within(1000000.1f, 1000000.0f, 0.0f, 1e-5f));
    }

    // error accumulates with the length of the chain: a longer dot product drifts further
    {
        // the same sum accumulated in two orders, over a chain long enough to disagree
        const size_t n = 1000;
        Tensor row{Shape({1, n})};
        Tensor column{Shape({n, 1})};
        for (size_t i = 0; i < n; ++i)
        {
            row.data()[i] = 1.0f / static_cast<float>(i + 1);
            column.data()[i] = 1.0f;
        }

        const float forward_order = matmul(row, column).at({0, 0});

        float reverse_order = 0.0f;
        for (size_t i = n; i-- > 0;)
        {
            reverse_order += row.data()[i];
        }

        // mathematically identical, numerically not — but within a sane tolerance
        CHECK(forward_order != reverse_order);
        CHECK(within(forward_order, reverse_order, 1e-5f, 1e-7f));
        CHECK(!within(forward_order, reverse_order, 0.0f, 0.0f));
    }

    // what a tolerance can and cannot distinguish
    {
        // rounding: passes at any sane tolerance
        CHECK(within(0.83000004f, 0.83f, 1e-5f, 1e-8f));

        // a transposed weight: wrong by orders of magnitude, and no tolerance hides it
        CHECK(!within(-4.21f, 0.83f, 1e-5f, 1e-8f));
        CHECK(!within(-4.21f, 0.83f, 1e-2f, 1e-2f));
        CHECK(!within(-4.21f, 0.83f, 1.0f, 1.0f));   // even at 100%

        // real bugs are not subtle numerically; they are subtle behaviourally
        CHECK(std::fabs(-4.21f - 0.83f) > 1000.0f * std::fabs(0.83000004f - 0.83f));
    }

    // zeros: the relative term vanishes and only atol applies
    {
        CHECK(!within(1e-7f, 0.0f, 1e-5f, 0.0f));    // rtol * 0 == 0, nothing passes
        CHECK(within(1e-7f, 0.0f, 1e-5f, 1e-6f));
        CHECK(within(0.0f, 0.0f, 0.0f, 0.0f));
    }

    // NaN never compares equal, to anything, including itself
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        CHECK(!within(nan, 1.0f, 1.0f, 1.0f));
        CHECK(!within(nan, nan, 1.0f, 1.0f));        // the comparison itself is false
        CHECK(!(nan == nan));

        // Infinities are a trap, and the tolerance rule gets both cases exactly backwards:
        //   inf vs -inf:  |inf - (-inf)| = inf,  threshold atol + rtol*inf = inf,
        //                 inf <= inf is true   -> declared equal, which is wrong
        //   inf vs  inf:  |inf - inf| = NaN,    NaN <= anything is false
        //                                      -> declared different, also wrong
        const float infinity = std::numeric_limits<float>::infinity();
        CHECK(within(infinity, -infinity, 1.0f, 1.0f));    // <- wrong, from the rule alone
        CHECK(!within(infinity, infinity, 1.0f, 1.0f));    // <- also wrong

        // T2 must therefore compare non-finite values before applying the tolerance:
        // equal infinities pass, opposite ones fail, and any NaN fails.
        CHECK(infinity == infinity);
        CHECK(!(infinity == -infinity));
    }

    // the tolerances the later epics will use, as a table that has to stay sane
    {
        // one op on small inputs: tight
        CHECK(within(19.000002f, 19.0f, 1e-6f, 1e-7f));
        // 28 layers end to end: loose, and still nowhere near hiding a real bug
        CHECK(within(0.8310f, 0.8300f, 1e-2f, 1e-4f));
        CHECK(!within(-0.83f, 0.83f, 1e-2f, 1e-4f));

        // and the criterion that sidesteps tolerance entirely: argmax
        const Tensor logits = filled(Shape({4}), {0.1f, 0.9f, 0.3f, 0.2f});
        const Tensor drifted = filled(Shape({4}), {0.1001f, 0.8998f, 0.3002f, 0.1999f});
        size_t best_of_logits = 0;
        size_t best_of_drifted = 0;
        for (size_t i = 1; i < 4; ++i)
        {
            best_of_logits = logits.at({i}) > logits.at({best_of_logits}) ? i : best_of_logits;
            best_of_drifted = drifted.at({i}) > drifted.at({best_of_drifted}) ? i : best_of_drifted;
        }
        CHECK_EQ(best_of_logits, best_of_drifted);   // the top-1 token survives the drift
    }

    return VEDA_TEST_SUMMARY("ToleranceTest");
}
