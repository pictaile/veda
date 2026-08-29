// E2.S2.T4 — matmul (2-D)

#include "Matmul.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <stdexcept>
#include <string>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::ops::matmul;

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
} // namespace

int main()
{
    // the hand-computed case, verified on paper before the code was written:
    //   [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
    {
        const Tensor a = filled(Shape({2, 2}), {1, 2, 3, 4});
        const Tensor b = filled(Shape({2, 2}), {5, 6, 7, 8});
        const Tensor c = matmul(a, b);

        CHECK_EQ(c.shape(), Shape({2, 2}));
        CHECK_EQ(c.at({0, 0}), 19.0f);   // 1*5 + 2*7
        CHECK_EQ(c.at({0, 1}), 22.0f);   // 1*6 + 2*8
        CHECK_EQ(c.at({1, 0}), 43.0f);   // 3*5 + 4*7
        CHECK_EQ(c.at({1, 1}), 50.0f);   // 3*6 + 4*8

        // it is not commutative: same shapes, different numbers
        const Tensor reversed = matmul(b, a);
        CHECK_EQ(reversed.at({0, 0}), 23.0f);   // 5*1 + 6*3
        CHECK(reversed.at({0, 0}) != c.at({0, 0}));

        // purity: fresh contiguous output, inputs untouched
        CHECK(c.is_contiguous());
        CHECK(c.storage() != a.storage());
        CHECK(c.storage() != b.storage());
        CHECK_EQ(a.at({0, 0}), 1.0f);
        CHECK_EQ(b.at({1, 1}), 8.0f);
    }

    // a row times a column collapses to one number
    {
        const Tensor row = filled(Shape({1, 3}), {1, 2, 3});
        const Tensor column = filled(Shape({3, 1}), {1, 0, 2});
        const Tensor c = matmul(row, column);
        CHECK_EQ(c.shape(), Shape({1, 1}));
        CHECK_EQ(c.at({0, 0}), 7.0f);

        // the other way round is an outer product: [3,1] x [1,3] -> [3,3]
        const Tensor outer = matmul(column, row);
        CHECK_EQ(outer.shape(), Shape({3, 3}));
        CHECK_EQ(outer.at({0, 2}), 3.0f);    // 1*3
        CHECK_EQ(outer.at({1, 1}), 0.0f);    // 0*2
        CHECK_EQ(outer.at({2, 0}), 2.0f);    // 2*1
    }

    // identity, both orders
    {
        const Tensor a = filled(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
        const Tensor right = matmul(a, identity(3));
        const Tensor left = matmul(identity(2), a);

        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                CHECK_EQ(right.at({i, j}), a.at({i, j}));
                CHECK_EQ(left.at({i, j}), a.at({i, j}));
            }
        }
    }

    // shapes: inner dimensions vanish, outer ones survive
    {
        const Tensor a{Shape({2, 3})};
        const Tensor b{Shape({3, 4})};
        CHECK_EQ(matmul(a, b).shape(), Shape({2, 4}));
        CHECK_EQ(matmul(Tensor{Shape({8, 1024})}, Tensor{Shape({1024, 32})}).shape(),
                 Shape({8, 32}));

        CHECK_THROWS_AS(matmul(a, Tensor{Shape({2, 3})}), std::invalid_argument);
        CHECK_THROWS_AS(matmul(a, Tensor{Shape({3})}), std::invalid_argument);   // rank 1
        CHECK_THROWS_AS(matmul(Tensor{Shape({})}, b), std::invalid_argument);    // rank 0
        // rank 3 against rank 2 is legal since T6 — this one still throws, on 2 against 3
        CHECK_THROWS_AS(matmul(Tensor{Shape({2, 2, 2})}, b), std::invalid_argument);

        try
        {
            (void)matmul(a, Tensor{Shape({2, 3})});
        }
        catch (const std::invalid_argument& error)
        {
            const std::string message = error.what();
            CHECK(message.find("(2, 3)") != std::string::npos);
            CHECK(message.find("3 against 2") != std::string::npos);
        }
    }

    // non-contiguous operands: the loop goes through the strides, not the buffer
    {
        const Tensor a = filled(Shape({2, 2}), {1, 2, 3, 4});
        const Tensor b = filled(Shape({2, 2}), {5, 6, 7, 8});

        // matmul(A, Bt) — the shape Q * Kt takes in attention
        const Tensor with_transposed_b = matmul(a, b.transpose(0, 1));
        CHECK_EQ(with_transposed_b.at({0, 0}), 17.0f);   // 1*5 + 2*6
        CHECK_EQ(with_transposed_b.at({0, 1}), 23.0f);   // 1*7 + 2*8
        CHECK_EQ(with_transposed_b.at({1, 0}), 39.0f);   // 3*5 + 4*6
        CHECK_EQ(with_transposed_b.at({1, 1}), 53.0f);   // 3*7 + 4*8

        // At * B
        const Tensor with_transposed_a = matmul(a.transpose(0, 1), b);
        CHECK_EQ(with_transposed_a.at({0, 0}), 26.0f);   // 1*5 + 3*7
        CHECK_EQ(with_transposed_a.at({1, 1}), 44.0f);   // 2*6 + 4*8

        // a sliced operand, with a non-zero offset
        const Tensor wide = filled(Shape({2, 4}), {1, 2, 3, 4, 5, 6, 7, 8});
        const Tensor window = wide.slice(1, 1, 2);       // [[2,3],[6,7]], strides (4,1), offset 1
        CHECK(!window.is_contiguous());
        const Tensor sliced_product = matmul(window, identity(2));
        CHECK_EQ(sliced_product.at({0, 0}), 2.0f);
        CHECK_EQ(sliced_product.at({0, 1}), 3.0f);
        CHECK_EQ(sliced_product.at({1, 0}), 6.0f);
        CHECK_EQ(sliced_product.at({1, 1}), 7.0f);
    }

    // a slightly larger case, checked against dot products computed independently
    {
        Tensor a{Shape({3, 4})};
        Tensor b{Shape({4, 2})};
        for (size_t i = 0; i < 12; ++i)
        {
            a.data()[i] = static_cast<float>(i + 1);
        }
        for (size_t i = 0; i < 8; ++i)
        {
            b.data()[i] = static_cast<float>((i % 3) + 1);
        }

        const Tensor c = matmul(a, b);
        CHECK_EQ(c.shape(), Shape({3, 2}));
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 2; ++j)
            {
                float expected = 0.0f;
                for (size_t k = 0; k < 4; ++k)
                {
                    expected += a.at({i, k}) * b.at({k, j});
                }
                CHECK_EQ(c.at({i, j}), expected);
            }
        }
    }

    // edge cases
    {
        // K == 0: the empty sum is 0, and the result is a zero-filled [M, N]
        const Tensor c = matmul(Tensor{Shape({2, 0})}, Tensor{Shape({0, 3})});
        CHECK_EQ(c.shape(), Shape({2, 3}));
        CHECK_EQ(c.numel(), size_t{6});
        CHECK_EQ(c.at({0, 0}), 0.0f);
        CHECK_EQ(c.at({1, 2}), 0.0f);

        // M or N equal to zero produces an empty result
        CHECK_EQ(matmul(Tensor{Shape({0, 3})}, Tensor{Shape({3, 4})}).numel(), size_t{0});
        CHECK_EQ(matmul(Tensor{Shape({2, 3})}, Tensor{Shape({3, 0})}).shape(), Shape({2, 0}));

        // 1x1 times 1x1
        CHECK_EQ(matmul(filled(Shape({1, 1}), {3}), filled(Shape({1, 1}), {4})).at({0, 0}), 12.0f);
    }

    return VEDA_TEST_SUMMARY("MatmulTest");
}
