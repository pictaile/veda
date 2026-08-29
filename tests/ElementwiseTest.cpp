// E2.S1.T2 — broadcast_shape, broadcast_to, add, mul

#include "Broadcast.h"
#include "Elementwise.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <stdexcept>
#include <string>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::ops::add;
using veda::ops::broadcast_shape;
using veda::ops::broadcast_to;
using veda::ops::mul;
using Strides = std::vector<size_t>;

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
    // --- broadcast_shape: the rule from T1 -----------------------------------------------------
    CHECK_EQ(broadcast_shape(Shape({2, 3}), Shape({2, 3})), Shape({2, 3}));
    CHECK_EQ(broadcast_shape(Shape({2, 3}), Shape({2, 1})), Shape({2, 3}));
    CHECK_EQ(broadcast_shape(Shape({2, 1}), Shape({2, 3})), Shape({2, 3}));
    CHECK_EQ(broadcast_shape(Shape({2, 3}), Shape({3})), Shape({2, 3}));
    CHECK_EQ(broadcast_shape(Shape({1, 8, 64}), Shape({8, 1})), Shape({1, 8, 64}));
    CHECK_EQ(broadcast_shape(Shape({}), Shape({2, 3})), Shape({2, 3}));
    CHECK_EQ(broadcast_shape(Shape({1, 16, 8, 8}), Shape({8, 8})), Shape({1, 16, 8, 8}));

    CHECK_THROWS_AS(broadcast_shape(Shape({2, 3}), Shape({3, 2})), std::invalid_argument);
    CHECK_THROWS_AS(broadcast_shape(Shape({2, 3}), Shape({4})), std::invalid_argument);
    try
    {
        (void)broadcast_shape(Shape({2, 3}), Shape({3, 2}));
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message = error.what();
        CHECK(message.find("(2, 3)") != std::string::npos);
        CHECK(message.find("(3, 2)") != std::string::npos);
    }

    // --- broadcast_to: a stride-0 view, never a copy -------------------------------------------
    {
        const Tensor bias = filled(Shape({3}), {7.0f, 8.0f, 9.0f});
        const Tensor spread = broadcast_to(bias, Shape({2, 3}));

        CHECK_EQ(spread.shape(), Shape({2, 3}));
        CHECK_EQ(spread.strides(), Strides({0, 1}));
        CHECK_EQ(spread.at({0, 1}), spread.at({1, 1}));      // both rows are the same three floats
        CHECK_EQ(spread.at({1, 2}), 9.0f);
        CHECK(spread.storage() == bias.storage());           // nothing was copied
        CHECK_EQ(spread.storage()->size(), size_t{3});
        CHECK(!spread.is_contiguous());

        // a per-token scale: the zero lands on the last axis instead
        const Tensor scale = filled(Shape({2, 1}), {10.0f, 20.0f});
        const Tensor spread_scale = broadcast_to(scale, Shape({2, 3}));
        CHECK_EQ(spread_scale.strides(), Strides({1, 0}));
        CHECK_EQ(spread_scale.at({0, 2}), 10.0f);
        CHECK_EQ(spread_scale.at({1, 0}), 20.0f);

        CHECK_THROWS_AS(broadcast_to(bias, Shape({2, 4})), std::invalid_argument);
        CHECK_THROWS_AS(broadcast_to(filled(Shape({2, 3}), {1, 2, 3, 4, 5, 6}), Shape({3})),
                        std::invalid_argument);
    }

    // --- add and mul on hand-computed 2x2 ------------------------------------------------------
    {
        const Tensor a = filled(Shape({2, 2}), {1, 2, 3, 4});
        const Tensor b = filled(Shape({2, 2}), {10, 20, 30, 40});

        const Tensor sum = add(a, b);
        CHECK_EQ(sum.shape(), Shape({2, 2}));
        CHECK_EQ(sum.at({0, 0}), 11.0f);
        CHECK_EQ(sum.at({0, 1}), 22.0f);
        CHECK_EQ(sum.at({1, 0}), 33.0f);
        CHECK_EQ(sum.at({1, 1}), 44.0f);

        const Tensor product = mul(a, b);
        CHECK_EQ(product.at({0, 0}), 10.0f);
        CHECK_EQ(product.at({0, 1}), 40.0f);
        CHECK_EQ(product.at({1, 0}), 90.0f);
        CHECK_EQ(product.at({1, 1}), 160.0f);

        // the output is fresh, contiguous, and independent of both inputs
        CHECK(sum.is_contiguous());
        CHECK(sum.storage() != a.storage());
        CHECK(sum.storage() != b.storage());

        // neither input was touched — checked, not merely intended
        CHECK_EQ(a.at({0, 0}), 1.0f);
        CHECK_EQ(a.at({1, 1}), 4.0f);
        CHECK_EQ(b.at({0, 0}), 10.0f);
        CHECK_EQ(b.at({1, 1}), 40.0f);
    }

    // --- broadcasting inside the ops -----------------------------------------------------------
    {
        // [2,3] * [2,1] — the RMSNorm shape
        const Tensor x = filled(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
        const Tensor scale = filled(Shape({2, 1}), {10, 20});
        const Tensor scaled = mul(x, scale);

        CHECK_EQ(scaled.shape(), Shape({2, 3}));
        CHECK_EQ(scaled.at({0, 0}), 10.0f);
        CHECK_EQ(scaled.at({0, 2}), 30.0f);
        CHECK_EQ(scaled.at({1, 0}), 80.0f);
        CHECK_EQ(scaled.at({1, 2}), 120.0f);

        // [2,3] + [3] — the Linear bias shape
        const Tensor bias = filled(Shape({3}), {100, 200, 300});
        const Tensor biased = add(x, bias);
        CHECK_EQ(biased.at({0, 0}), 101.0f);
        CHECK_EQ(biased.at({1, 2}), 306.0f);

        // the operands may be given either way round
        CHECK_EQ(add(bias, x).at({1, 2}), 306.0f);
        CHECK_EQ(mul(scale, x).at({1, 0}), 80.0f);

        // a rank-0 tensor against a rank-3 one
        const Tensor scalar_tensor = filled(Shape({}), {2.0f});
        const Tensor cube = filled(Shape({1, 2, 2}), {1, 2, 3, 4});
        const Tensor doubled = mul(cube, scalar_tensor);
        CHECK_EQ(doubled.shape(), Shape({1, 2, 2}));
        CHECK_EQ(doubled.at({0, 1, 1}), 8.0f);

        CHECK_THROWS_AS(add(x, filled(Shape({3, 2}), {1, 2, 3, 4, 5, 6})), std::invalid_argument);
    }

    // --- scalar variants ------------------------------------------------------------------------
    {
        const Tensor a = filled(Shape({2, 2}), {1, 2, 3, 4});

        const Tensor halved = mul(a, 0.5f);
        CHECK_EQ(halved.at({0, 0}), 0.5f);
        CHECK_EQ(halved.at({1, 1}), 2.0f);

        const Tensor shifted = add(a, 10.0f);
        CHECK_EQ(shifted.at({0, 0}), 11.0f);
        CHECK_EQ(shifted.at({1, 1}), 14.0f);

        // the attention scale, as it will be written in E7: mul(scores, 1/sqrt(Dh))
        const Tensor scores = filled(Shape({2, 2}), {4, 8, 12, 16});
        const Tensor scaled = mul(scores, 1.0f / 4.0f);
        CHECK_EQ(scaled.at({0, 1}), 2.0f);
        CHECK_EQ(scaled.at({1, 1}), 4.0f);

        CHECK_EQ(a.at({0, 0}), 1.0f);   // input untouched
    }

    // --- aliasing: inputs may be views over the same storage ------------------------------------
    {
        Tensor x = filled(Shape({2, 2}), {1, 2, 3, 4});

        const Tensor doubled = add(x, x);
        CHECK_EQ(doubled.at({0, 1}), 4.0f);
        CHECK_EQ(doubled.at({1, 1}), 8.0f);
        CHECK_EQ(doubled.at({0, 0}), mul(x, 2.0f).at({0, 0}));
        CHECK_EQ(doubled.at({1, 0}), mul(x, 2.0f).at({1, 0}));
        CHECK_EQ(x.at({1, 1}), 4.0f);   // still the original

        // x + xT on a square tensor: reading the same storage through two different rules
        const Tensor symmetric = add(x, x.transpose(0, 1));
        CHECK_EQ(symmetric.at({0, 1}), 5.0f);   // 2 + 3
        CHECK_EQ(symmetric.at({1, 0}), 5.0f);
        CHECK_EQ(symmetric.at({0, 0}), 2.0f);
    }

    // --- non-contiguous inputs -------------------------------------------------------------------
    {
        const Tensor a = filled(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
        const Tensor transposed = a.transpose(0, 1);            // (3,2), strides (1,3)
        CHECK(!transposed.is_contiguous());

        const Tensor b = filled(Shape({3, 2}), {10, 20, 30, 40, 50, 60});
        const Tensor sum = add(transposed, b);
        CHECK_EQ(sum.shape(), Shape({3, 2}));
        CHECK_EQ(sum.at({0, 0}), 11.0f);    // 1 + 10
        CHECK_EQ(sum.at({0, 1}), 24.0f);    // 4 + 20
        CHECK_EQ(sum.at({2, 1}), 66.0f);    // 6 + 60
        CHECK(sum.is_contiguous());          // the result always is

        // a sliced input, with a non-zero offset
        const Tensor window = a.slice(1, 1, 2);                 // (2,2), offset 1, strides (3,1)
        const Tensor scaled = mul(window, 2.0f);
        CHECK_EQ(scaled.at({0, 0}), 4.0f);
        CHECK_EQ(scaled.at({1, 1}), 12.0f);
        CHECK(scaled.is_contiguous());
    }

    // --- edge cases -------------------------------------------------------------------------------
    {
        // rank 0 against rank 0
        const Tensor s = filled(Shape({}), {3.0f});
        CHECK_EQ(add(s, s).at({}), 6.0f);
        CHECK_EQ(mul(s, 4.0f).at({}), 12.0f);
        CHECK_EQ(add(s, s).shape().rank(), size_t{0});

        // a zero-element tensor stays empty and allocates nothing to describe
        const Tensor empty{Shape({2, 0, 3})};
        CHECK_EQ(add(empty, empty).numel(), size_t{0});
        CHECK_EQ(mul(empty, 2.0f).shape(), Shape({2, 0, 3}));

        // a dimension of 1 on both sides stays 1
        CHECK_EQ(broadcast_shape(Shape({1, 3}), Shape({1, 1})), Shape({1, 3}));

        // a zero extent against 1 stays zero — taking the larger would claim elements that the
        // operand does not have, and the op would read past the end of an empty storage
        CHECK_EQ(broadcast_shape(Shape({0, 3}), Shape({1, 3})), Shape({0, 3}));
        CHECK_EQ(broadcast_shape(Shape({0, 2, 3}), Shape({})), Shape({0, 2, 3}));
        CHECK_EQ(add(Tensor{Shape({0, 3})}, Tensor{Shape({1, 3})}).numel(), size_t{0});
        CHECK_EQ(mul(Tensor{Shape({2, 0})}, Tensor{Shape({2, 1})}).shape(), Shape({2, 0}));
    }

    return VEDA_TEST_SUMMARY("ElementwiseTest");
}
