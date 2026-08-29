// E2.S2.T5 — matmul_nt, and the AD4 equivalence that guards it

#include "Matmul.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::ops::matmul;
using veda::ops::matmul_nt;

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

Tensor random_tensor(const Shape& shape, std::mt19937& engine)
{
    std::uniform_real_distribution<float> values(-2.0f, 2.0f);
    Tensor t{shape};
    for (size_t i = 0; i < shape.size(); ++i)
    {
        t.data()[i] = values(engine);
    }
    return t;
}
} // namespace

int main()
{
    // the hand-computed case: B is T4's matrix already transposed, so the answer is the same
    //   A  = [[1,2],[3,4]]      B = [[5,7],[6,8]]
    //   C[i,j] = row i of A against row j of B
    {
        const Tensor a = filled(Shape({2, 2}), {1, 2, 3, 4});
        const Tensor b = filled(Shape({2, 2}), {5, 7, 6, 8});
        const Tensor c = matmul_nt(a, b);

        CHECK_EQ(c.shape(), Shape({2, 2}));
        CHECK_EQ(c.at({0, 0}), 19.0f);   // 1*5 + 2*7
        CHECK_EQ(c.at({0, 1}), 22.0f);   // 1*6 + 2*8
        CHECK_EQ(c.at({1, 0}), 43.0f);   // 3*5 + 4*7
        CHECK_EQ(c.at({1, 1}), 50.0f);   // 3*6 + 4*8

        // and it is exactly what plain matmul gives for the transposed B
        const Tensor plain = matmul(a, filled(Shape({2, 2}), {5, 6, 7, 8}));
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 2; ++j)
            {
                CHECK_EQ(c.at({i, j}), plain.at({i, j}));
            }
        }

        CHECK(c.is_contiguous());
        CHECK(c.storage() != a.storage());
        CHECK_EQ(a.at({0, 0}), 1.0f);   // inputs untouched
        CHECK_EQ(b.at({1, 1}), 8.0f);
    }

    // *** the AD4 test: the reason this function exists ***
    // matmul_nt(A, B) == matmul(A, B.transpose(0, 1)) — on random values, over several shapes.
    // The summation order is identical in both paths, so this is exact equality, not a tolerance.
    {
        std::mt19937 engine(20260829);
        const std::vector<std::vector<size_t>> cases = {
            {2, 3, 4}, {1, 1, 1}, {5, 2, 1}, {1, 7, 3}, {4, 4, 4}, {8, 16, 3}, {3, 1, 5},
        };

        for (const std::vector<size_t>& dims : cases)
        {
            const size_t m = dims[0];
            const size_t k = dims[1];
            const size_t n = dims[2];

            const Tensor a = random_tensor(Shape({m, k}), engine);
            const Tensor b = random_tensor(Shape({n, k}), engine);   // [N, K], the HF layout

            const Tensor from_nt = matmul_nt(a, b);
            const Tensor from_transpose = matmul(a, b.transpose(0, 1));

            CHECK_EQ(from_nt.shape(), Shape({m, n}));
            CHECK_EQ(from_transpose.shape(), from_nt.shape());

            bool identical = true;
            for (size_t i = 0; i < m && identical; ++i)
            {
                for (size_t j = 0; j < n; ++j)
                {
                    if (from_nt.at({i, j}) != from_transpose.at({i, j}))
                    {
                        identical = false;
                        break;
                    }
                }
            }
            CHECK(identical);
        }
    }

    // shapes: the shared dimension is now the LAST of both operands
    {
        CHECK_EQ(matmul_nt(Tensor{Shape({2, 3})}, Tensor{Shape({4, 3})}).shape(), Shape({2, 4}));
        CHECK_EQ(matmul_nt(Tensor{Shape({8, 1024})}, Tensor{Shape({32, 1024})}).shape(),
                 Shape({8, 32}));

        // a Linear layer, in the layout the file actually has: [T, in] x [out, in] -> [T, out]
        CHECK_EQ(matmul_nt(Tensor{Shape({8, 1024})}, Tensor{Shape({3072, 1024})}).shape(),
                 Shape({8, 3072}));

        // attention's Q * Kt, with K used directly as [T, Dh]
        CHECK_EQ(matmul_nt(Tensor{Shape({8, 64})}, Tensor{Shape({8, 64})}).shape(), Shape({8, 8}));

        CHECK_THROWS_AS(matmul_nt(Tensor{Shape({2, 3})}, Tensor{Shape({4, 2})}),
                        std::invalid_argument);
        CHECK_THROWS_AS(matmul_nt(Tensor{Shape({2, 3})}, Tensor{Shape({3, 4})}),
                        std::invalid_argument);   // the plain-matmul layout is refused here
        CHECK_THROWS_AS(matmul_nt(Tensor{Shape({2, 3})}, Tensor{Shape({3})}), std::invalid_argument);
        CHECK_THROWS_AS(matmul_nt(Tensor{Shape({2})}, Tensor{Shape({2, 2})}), std::invalid_argument);

        // rank 3 is no longer a refusal: since T6 the leading dimensions are batch axes
        CHECK_EQ(matmul_nt(Tensor{Shape({2, 2, 2})}, Tensor{Shape({2, 2})}).shape(),
                 Shape({2, 2, 2}));

        try
        {
            (void)matmul_nt(Tensor{Shape({2, 3})}, Tensor{Shape({4, 2})});
        }
        catch (const std::invalid_argument& error)
        {
            const std::string message = error.what();
            CHECK(message.find("matmul_nt") != std::string::npos);
            CHECK(message.find("(2, 3)") != std::string::npos);
            CHECK(message.find("(4, 2)") != std::string::npos);
            CHECK(message.find("3 against 2") != std::string::npos);
        }
    }

    // identity: with B = I the two functions agree trivially, since I is its own transpose
    {
        const Tensor a = filled(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
        const Tensor through_nt = matmul_nt(a, identity(3));
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                CHECK_EQ(through_nt.at({i, j}), a.at({i, j}));
            }
        }
    }

    // non-contiguous operands on either side
    {
        const Tensor a = filled(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
        const Tensor b = filled(Shape({2, 3}), {1, 0, 2, 0, 1, 1});

        // A transposed: (3,2) x (2,2)t -> handled through strides
        const Tensor at = a.transpose(0, 1);
        CHECK(!at.is_contiguous());
        CHECK_EQ(matmul_nt(at, filled(Shape({4, 2}), {1, 0, 0, 1, 1, 1, 2, 3})).shape(),
                 Shape({3, 4}));

        // B transposed: a [3,2] view of a [2,3] tensor used as the [N,K] operand
        const Tensor bt = b.transpose(0, 1);   // (3,2)
        const Tensor c = matmul_nt(filled(Shape({1, 2}), {1, 1}), bt);
        CHECK_EQ(c.shape(), Shape({1, 3}));
        CHECK_EQ(c.at({0, 0}), 1.0f);   // 1*1 + 1*0
        CHECK_EQ(c.at({0, 1}), 1.0f);   // 1*0 + 1*1
        CHECK_EQ(c.at({0, 2}), 3.0f);   // 1*2 + 1*1

        // a sliced B keeps working
        const Tensor sliced = b.slice(1, 1, 2);   // (2,2), offset 1, strides (3,1)
        CHECK_EQ(matmul_nt(filled(Shape({1, 2}), {1, 0}), sliced).at({0, 0}), 0.0f);   // row [0,2]
        CHECK_EQ(matmul_nt(filled(Shape({1, 2}), {0, 1}), sliced).at({0, 1}), 1.0f);   // row [1,1]
    }

    // edge cases
    {
        // N == 1: a single output feature
        CHECK_EQ(matmul_nt(filled(Shape({2, 2}), {1, 2, 3, 4}), filled(Shape({1, 2}), {1, 1}))
                     .at({1, 0}),
                 7.0f);   // 3 + 4

        // K == 0: the empty sum is 0
        const Tensor empty_shared = matmul_nt(Tensor{Shape({2, 0})}, Tensor{Shape({3, 0})});
        CHECK_EQ(empty_shared.shape(), Shape({2, 3}));
        CHECK_EQ(empty_shared.at({1, 2}), 0.0f);

        // an empty output
        CHECK_EQ(matmul_nt(Tensor{Shape({0, 3})}, Tensor{Shape({4, 3})}).numel(), size_t{0});
    }

    return VEDA_TEST_SUMMARY("MatmulNtTest");
}
