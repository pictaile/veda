// E2.S3.T8 — softmax along an arbitrary dimension

#include "Shape.h"
#include "Softmax.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
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

float lane_sum(const Tensor& t, size_t row)
{
    float sum = 0.0f;
    for (size_t j = 0; j < t.shape()[1]; ++j)
    {
        sum += t.at({row, j});
    }
    return sum;
}

const float infinity = std::numeric_limits<float>::infinity();
} // namespace

int main()
{
    // equal inputs give a uniform distribution
    {
        const Tensor p = softmax(filled(Shape({3}), {1, 1, 1}), 0);
        CHECK_EQ(p.shape(), Shape({3}));
        CHECK_NEAR(p.at({0}), 1.0 / 3.0, 1e-6);
        CHECK_NEAR(p.at({1}), 1.0 / 3.0, 1e-6);
        CHECK_NEAR(p.at({2}), 1.0 / 3.0, 1e-6);
        CHECK_NEAR(p.at({0}) + p.at({1}) + p.at({2}), 1.0, 1e-6);

        // only differences matter
        const Tensor same = softmax(filled(Shape({3}), {5, 5, 5}), 0);
        CHECK_NEAR(same.at({0}), 1.0 / 3.0, 1e-6);
    }

    // the worked example
    {
        const Tensor p = softmax(filled(Shape({3}), {2.0f, 1.0f, 0.1f}), 0);
        CHECK_NEAR(p.at({0}), 0.659, 1e-3);
        CHECK_NEAR(p.at({1}), 0.242, 1e-3);
        CHECK_NEAR(p.at({2}), 0.099, 1e-3);
        CHECK_NEAR(p.at({0}) + p.at({1}) + p.at({2}), 1.0, 1e-6);
        CHECK(p.at({0}) > p.at({1}) && p.at({1}) > p.at({2}));   // order preserved
        CHECK(p.is_contiguous());
    }

    // *** the stability case, the reason the max subtraction is in the code ***
    {
        const Tensor big = softmax(filled(Shape({2}), {1000.0f, 1001.0f}), 0);
        CHECK(!std::isnan(big.at({0})));
        CHECK(!std::isnan(big.at({1})));
        CHECK_NEAR(big.at({0}), 0.2689, 1e-4);
        CHECK_NEAR(big.at({1}), 0.7311, 1e-4);
        CHECK_NEAR(big.at({0}) + big.at({1}), 1.0, 1e-6);

        // and it is the same distribution as the small-numbered case
        const Tensor small = softmax(filled(Shape({2}), {1.0f, 2.0f}), 0);
        CHECK_NEAR(big.at({0}), small.at({0}), 1e-6);
        CHECK_NEAR(big.at({1}), small.at({1}), 1e-6);

        // shift invariance, over a range of shifts
        for (const float shift : {-500.0f, -20.0f, 0.0f, 37.0f, 700.0f})
        {
            const Tensor shifted = softmax(filled(Shape({2}), {1.0f + shift, 2.0f + shift}), 0);
            CHECK_NEAR(shifted.at({0}), small.at({0}), 1e-6);
        }

        // an extreme spread saturates rather than breaking
        const Tensor extreme = softmax(filled(Shape({2}), {-1000.0f, 1000.0f}), 0);
        CHECK_NEAR(extreme.at({0}), 0.0, 1e-9);
        CHECK_NEAR(extreme.at({1}), 1.0, 1e-9);
    }

    // along either axis of the same 2x2 — both are valid distributions, only one is the answer
    {
        const Tensor x = filled(Shape({2, 2}), {1, 2, 3, 4});

        const Tensor by_row = softmax(x, 1);
        CHECK_NEAR(by_row.at({0, 0}), 0.2689, 1e-4);
        CHECK_NEAR(by_row.at({0, 1}), 0.7311, 1e-4);
        CHECK_NEAR(by_row.at({1, 0}), 0.2689, 1e-4);
        CHECK_NEAR(lane_sum(by_row, 0), 1.0, 1e-6);
        CHECK_NEAR(lane_sum(by_row, 1), 1.0, 1e-6);

        const Tensor by_column = softmax(x, 0);
        CHECK_NEAR(by_column.at({0, 0}), 0.1192, 1e-4);
        CHECK_NEAR(by_column.at({1, 0}), 0.8808, 1e-4);
        CHECK_NEAR(by_column.at({0, 0}) + by_column.at({1, 0}), 1.0, 1e-6);
        CHECK_NEAR(by_column.at({0, 1}) + by_column.at({1, 1}), 1.0, 1e-6);

        // the input is untouched by either call
        CHECK_EQ(x.at({0, 0}), 1.0f);
        CHECK_EQ(x.at({1, 1}), 4.0f);
    }

    // masking, exactly as E7 will use it
    {
        const Tensor masked = softmax(filled(Shape({3}), {1.0f, -infinity, 2.0f}), 0);
        CHECK_EQ(masked.at({1}), 0.0f);   // exactly zero, not merely small
        CHECK_NEAR(masked.at({0}) + masked.at({1}) + masked.at({2}), 1.0, 1e-6);

        // the remaining positions are distributed as if the masked one were absent
        const Tensor without = softmax(filled(Shape({2}), {1.0f, 2.0f}), 0);
        CHECK_NEAR(masked.at({0}), without.at({0}), 1e-6);
        CHECK_NEAR(masked.at({2}), without.at({1}), 1e-6);

        // a causal row: everything after position t masked out
        const Tensor causal_row =
            softmax(filled(Shape({4}), {0.5f, 1.5f, -infinity, -infinity}), 0);
        CHECK_EQ(causal_row.at({2}), 0.0f);
        CHECK_EQ(causal_row.at({3}), 0.0f);
        CHECK_NEAR(causal_row.at({0}) + causal_row.at({1}), 1.0, 1e-6);
    }

    // the attention shape: [B, H, T, T] along the last axis, every query row a distribution
    {
        Tensor scores{Shape({1, 2, 3, 3})};
        for (size_t i = 0; i < scores.numel(); ++i)
        {
            scores.data()[i] = static_cast<float>(i % 5);
        }

        const Tensor weights = softmax(scores, 3);
        CHECK_EQ(weights.shape(), Shape({1, 2, 3, 3}));
        CHECK(weights.is_contiguous());

        for (size_t h = 0; h < 2; ++h)
        {
            for (size_t t = 0; t < 3; ++t)
            {
                float sum = 0.0f;
                for (size_t j = 0; j < 3; ++j)
                {
                    sum += weights.at({0, h, t, j});
                    CHECK(weights.at({0, h, t, j}) > 0.0f);
                }
                CHECK_NEAR(sum, 1.0, 1e-6);
            }
        }
    }

    // non-contiguous input: a transposed score matrix arrives here constantly
    {
        const Tensor x = filled(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
        const Tensor transposed = x.transpose(0, 1);   // (3,2), strides (1,3)
        CHECK(!transposed.is_contiguous());

        const Tensor p = softmax(transposed, 1);
        CHECK_EQ(p.shape(), Shape({3, 2}));
        CHECK(p.is_contiguous());
        // row 0 of the transposed view is [1, 4]
        CHECK_NEAR(p.at({0, 0}), softmax(filled(Shape({2}), {1, 4}), 0).at({0}), 1e-6);
        CHECK_NEAR(p.at({0, 0}) + p.at({0, 1}), 1.0, 1e-6);
        CHECK_NEAR(p.at({2, 0}) + p.at({2, 1}), 1.0, 1e-6);

        // a sliced input, with a non-zero offset
        const Tensor window = x.slice(1, 1, 2);   // [[2,3],[5,6]], offset 1
        const Tensor sliced = softmax(window, 1);
        CHECK_NEAR(sliced.at({0, 0}), 0.2689, 1e-4);
        CHECK_NEAR(sliced.at({1, 0}) + sliced.at({1, 1}), 1.0, 1e-6);
    }

    // edge cases
    {
        // an extent of 1 gives exactly 1.0 — exp(0)/exp(0), not 0.9999
        const Tensor single = softmax(filled(Shape({3, 1}), {5, -2, 100}), 1);
        CHECK_EQ(single.at({0, 0}), 1.0f);
        CHECK_EQ(single.at({1, 0}), 1.0f);
        CHECK_EQ(single.at({2, 0}), 1.0f);

        // an extent of 0 leaves an empty result of the same shape
        const Tensor empty = softmax(Tensor{Shape({2, 0})}, 1);
        CHECK_EQ(empty.shape(), Shape({2, 0}));
        CHECK_EQ(empty.numel(), size_t{0});

        // a rank-0 tensor has no dimension to take softmax along
        CHECK_THROWS_AS(softmax(Tensor{Shape({})}, 0), std::invalid_argument);
        CHECK_THROWS_AS(softmax(Tensor{Shape({2, 3})}, 2), std::invalid_argument);
        try
        {
            (void)softmax(Tensor{Shape({2, 3})}, 7);
        }
        catch (const std::invalid_argument& error)
        {
            const std::string message = error.what();
            CHECK(message.find("(2, 3)") != std::string::npos);
            CHECK(message.find("7") != std::string::npos);
        }

        // a lane of identical values, at any magnitude
        const Tensor flat = softmax(filled(Shape({4}), {7, 7, 7, 7}), 0);
        CHECK_NEAR(flat.at({0}), 0.25, 1e-6);
        CHECK_NEAR(flat.at({3}), 0.25, 1e-6);
    }

    return VEDA_TEST_SUMMARY("SoftmaxTest");
}
