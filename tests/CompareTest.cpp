// E3.S1.T2 — compare_close and assert_close

#include "Compare.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::testing::assert_close;
using veda::testing::compare_close;
using veda::testing::Difference;

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

const float nan = std::numeric_limits<float>::quiet_NaN();
const float infinity = std::numeric_limits<float>::infinity();
} // namespace

int main()
{
    // identical tensors agree
    {
        const Tensor a = filled(Shape({2, 2}), {1, 2, 3, 4});
        const Tensor b = filled(Shape({2, 2}), {1, 2, 3, 4});
        const Difference d = compare_close(a, b, 1e-5f, 1e-8f);

        CHECK(d.equal);
        CHECK_EQ(d.mismatches, size_t{0});
        CHECK_EQ(d.total, size_t{4});
        CHECK(d.message.empty());
    }

    // one element differing is reported at its own index
    {
        const Tensor a = filled(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
        const Tensor b = filled(Shape({2, 3}), {1, 2, 3, 4, 9, 6});
        const Difference d = compare_close(a, b, 1e-5f, 1e-8f);

        CHECK(!d.equal);
        CHECK_EQ(d.mismatches, size_t{1});
        CHECK_EQ(d.index, std::vector<size_t>({1, 1}));
        CHECK_EQ(d.actual, 5.0f);
        CHECK_EQ(d.expected, 9.0f);
        CHECK_EQ(d.absolute, 4.0f);
        CHECK(d.message.find("(1, 1)") != std::string::npos);
        CHECK(d.message.find("1 of 6 elements differ") != std::string::npos);
    }

    // several differing: the WORST is reported, not the first, and all are counted
    {
        const Tensor a = filled(Shape({4}), {1.0f, 2.0f, 3.0f, 4.0f});
        const Tensor b = filled(Shape({4}), {1.1f, 2.0f, 9.0f, 4.05f});
        const Difference d = compare_close(a, b, 1e-5f, 1e-8f);

        CHECK(!d.equal);
        CHECK_EQ(d.mismatches, size_t{3});
        CHECK_EQ(d.index, std::vector<size_t>({2}));   // index 0 fails first; index 2 fails worst
        CHECK_EQ(d.expected, 9.0f);
        CHECK(d.message.find("3 of 4 elements differ") != std::string::npos);
    }

    // the mismatch count is what distinguishes the kind of bug
    {
        const Tensor a = filled(Shape({2, 2}), {1, 2, 3, 4});

        // everything wrong — a transposed operand or a sign flip
        const Tensor all_flipped = filled(Shape({2, 2}), {-1, -2, -3, -4});
        CHECK_EQ(compare_close(a, all_flipped, 1e-5f, 1e-8f).mismatches, size_t{4});

        // one wrong — an indexing bug
        const Tensor one_off = filled(Shape({2, 2}), {1, 2, 3, 4.5f});
        CHECK_EQ(compare_close(a, one_off, 1e-5f, 1e-8f).mismatches, size_t{1});
    }

    // shape mismatch is reported distinctly, without comparing values
    {
        const Difference d = compare_close(Tensor{Shape({2, 3})}, Tensor{Shape({3, 2})}, 1e-5f, 1e-8f);
        CHECK(!d.equal);
        CHECK_EQ(d.mismatches, size_t{0});          // no element comparison happened
        CHECK(d.message.find("differ in shape") != std::string::npos);
        CHECK(d.message.find("(2, 3)") != std::string::npos);
        CHECK(d.message.find("(3, 2)") != std::string::npos);

        CHECK(!compare_close(Tensor{Shape({6})}, Tensor{Shape({2, 3})}, 1e-5f, 1e-8f).equal);
    }

    // the tolerance rule, and the regimes each term covers
    {
        // rounding noise passes
        CHECK(compare_close(filled(Shape({1}), {1.0000001f}), filled(Shape({1}), {1.0f}), 1e-5f, 1e-8f)
                  .equal);

        // 4.0001 vs 4.0: fails at rtol 1e-5, passes at 1e-3
        const Tensor a = filled(Shape({1}), {4.0001f});
        const Tensor b = filled(Shape({1}), {4.0f});
        CHECK(!compare_close(a, b, 1e-5f, 1e-8f).equal);
        CHECK(compare_close(a, b, 1e-3f, 1e-8f).equal);

        // near zero, only atol can save the comparison
        const Tensor tiny = filled(Shape({1}), {2e-9f});
        const Tensor tinier = filled(Shape({1}), {1e-9f});
        CHECK(!compare_close(tiny, tinier, 1e-5f, 0.0f).equal);
        CHECK(compare_close(tiny, tinier, 1e-5f, 1e-8f).equal);

        // an expected of exactly zero leaves only atol in play
        CHECK(!compare_close(filled(Shape({1}), {1e-7f}), filled(Shape({1}), {0.0f}), 1e-5f, 0.0f)
                   .equal);
        CHECK(compare_close(filled(Shape({1}), {1e-7f}), filled(Shape({1}), {0.0f}), 1e-5f, 1e-6f)
                  .equal);
    }

    // non-finite values, which the plain rule gets backwards in both directions
    {
        // any NaN fails, including NaN against NaN
        CHECK(!compare_close(filled(Shape({1}), {nan}), filled(Shape({1}), {0.0f}), 1.0f, 1.0f).equal);
        CHECK(!compare_close(filled(Shape({1}), {0.0f}), filled(Shape({1}), {nan}), 1.0f, 1.0f).equal);
        CHECK(!compare_close(filled(Shape({1}), {nan}), filled(Shape({1}), {nan}), 1.0f, 1.0f).equal);

        // equal infinities pass; opposite ones do not
        CHECK(compare_close(filled(Shape({1}), {infinity}), filled(Shape({1}), {infinity}), 0.0f, 0.0f)
                  .equal);
        CHECK(compare_close(filled(Shape({1}), {-infinity}), filled(Shape({1}), {-infinity}), 0.0f,
                            0.0f)
                  .equal);
        CHECK(!compare_close(filled(Shape({1}), {infinity}), filled(Shape({1}), {-infinity}), 1e9f,
                             1e9f)
                   .equal);
        CHECK(!compare_close(filled(Shape({1}), {infinity}), filled(Shape({1}), {1.0f}), 1e9f, 1e9f)
                   .equal);

        // a NaN among finite values is reported at its own index
        const Difference d = compare_close(filled(Shape({3}), {1.0f, nan, 3.0f}),
                                           filled(Shape({3}), {1.0f, 2.0f, 3.0f}), 1e-5f, 1e-8f);
        CHECK(!d.equal);
        CHECK_EQ(d.index, std::vector<size_t>({1}));
        CHECK(d.message.find("nan") != std::string::npos);
    }

    // non-contiguous operands: a reference tensor is often a view
    {
        const Tensor a = filled(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
        const Tensor transposed = a.transpose(0, 1);
        const Tensor expected = filled(Shape({3, 2}), {1, 4, 2, 5, 3, 6});
        CHECK(!transposed.is_contiguous());
        CHECK(compare_close(transposed, expected, 1e-6f, 1e-8f).equal);

        // and a sliced one, with a non-zero offset
        const Tensor window = a.slice(1, 1, 2);   // [[2,3],[5,6]]
        CHECK(compare_close(window, filled(Shape({2, 2}), {2, 3, 5, 6}), 1e-6f, 1e-8f).equal);
        CHECK(!compare_close(window, filled(Shape({2, 2}), {2, 3, 5, 7}), 1e-6f, 1e-8f).equal);
    }

    // the index has the rank of the tensors, so a failure names the head and position
    {
        Tensor actual{Shape({1, 2, 3, 3})};
        Tensor expected{Shape({1, 2, 3, 3})};
        actual.data()[1 * 9 + 2 * 3 + 1] = 5.0f;   // head 1, query 2, key 1

        const Difference d = compare_close(actual, expected, 1e-5f, 1e-8f);
        CHECK(!d.equal);
        CHECK_EQ(d.index, std::vector<size_t>({0, 1, 2, 1}));
        CHECK(d.message.find("(0, 1, 2, 1)") != std::string::npos);
        CHECK(d.message.find("(1, 2, 3, 3)") != std::string::npos);   // the shape, for context
    }

    // assert_close throws with the same message compare_close builds
    {
        const Tensor a = filled(Shape({2}), {1.0f, 2.0f});
        const Tensor b = filled(Shape({2}), {1.0f, 2.5f});

        assert_close(a, a, 1e-6f, 1e-8f);   // must not throw
        CHECK_THROWS_AS(assert_close(a, b, 1e-6f, 1e-8f), std::runtime_error);

        try
        {
            assert_close(a, b, 1e-6f, 1e-8f);
        }
        catch (const std::runtime_error& error)
        {
            const std::string message = error.what();
            CHECK(message == compare_close(a, b, 1e-6f, 1e-8f).message);
            CHECK(message.find("(1)") != std::string::npos);
            CHECK(message.find("1 of 2") != std::string::npos);
        }
    }

    // edge cases
    {
        // two empty tensors of the same shape agree
        CHECK(compare_close(Tensor{Shape({2, 0})}, Tensor{Shape({2, 0})}, 0.0f, 0.0f).equal);

        // rank-0 tensors
        CHECK(compare_close(filled(Shape({}), {1.0f}), filled(Shape({}), {1.0f}), 0.0f, 0.0f).equal);
        const Difference d =
            compare_close(filled(Shape({}), {1.0f}), filled(Shape({}), {2.0f}), 0.0f, 0.0f);
        CHECK(!d.equal);
        CHECK_EQ(d.index.size(), size_t{0});
        CHECK(d.message.find("()") != std::string::npos);

        // every element differing
        CHECK_EQ(compare_close(filled(Shape({3}), {1, 2, 3}), filled(Shape({3}), {4, 5, 6}), 0.0f,
                               0.0f)
                     .mismatches,
                 size_t{3});
    }

    return VEDA_TEST_SUMMARY("CompareTest");
}
