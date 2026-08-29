// E2.S2.T6 — batched matmul over leading dimensions

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
    // two stacked 2x2 products, both verifiable by hand
    //   A[0] = [[1,2],[3,4]]   B[0] = [[5,6],[7,8]]  ->  [[19,22],[43,50]]
    //   A[1] = identity        B[1] = [[9,1],[2,3]]  ->  B[1] unchanged
    {
        const Tensor a = filled(Shape({2, 2, 2}), {1, 2, 3, 4, 1, 0, 0, 1});
        const Tensor b = filled(Shape({2, 2, 2}), {5, 6, 7, 8, 9, 1, 2, 3});
        const Tensor c = matmul(a, b);

        CHECK_EQ(c.shape(), Shape({2, 2, 2}));
        CHECK_EQ(c.at({0, 0, 0}), 19.0f);
        CHECK_EQ(c.at({0, 0, 1}), 22.0f);
        CHECK_EQ(c.at({0, 1, 0}), 43.0f);
        CHECK_EQ(c.at({0, 1, 1}), 50.0f);
        CHECK_EQ(c.at({1, 0, 0}), 9.0f);
        CHECK_EQ(c.at({1, 0, 1}), 1.0f);
        CHECK_EQ(c.at({1, 1, 0}), 2.0f);
        CHECK_EQ(c.at({1, 1, 1}), 3.0f);

        CHECK(c.is_contiguous());
        CHECK_EQ(a.at({0, 0, 0}), 1.0f);   // inputs untouched
    }

    // every batch element equals the 2-D product of the corresponding slices — the rank-2 path
    // stays the tested reference for the batched one
    {
        std::mt19937 engine(20260830);
        const Tensor a = random_tensor(Shape({3, 4, 2}), engine);   // 3 matrices of [4,2]
        const Tensor b = random_tensor(Shape({3, 2, 5}), engine);   // 3 matrices of [2,5]
        const Tensor batched = matmul(a, b);

        CHECK_EQ(batched.shape(), Shape({3, 4, 5}));

        bool identical = true;
        for (size_t e = 0; e < 3 && identical; ++e)
        {
            // slice out one matrix and multiply it on its own
            const Tensor a_slice = a.slice(0, e, 1).reshape(Shape({4, 2}));
            const Tensor b_slice = b.slice(0, e, 1).reshape(Shape({2, 5}));
            const Tensor expected = matmul(a_slice, b_slice);

            for (size_t i = 0; i < 4 && identical; ++i)
            {
                for (size_t j = 0; j < 5; ++j)
                {
                    if (batched.at({e, i, j}) != expected.at({i, j}))
                    {
                        identical = false;
                        break;
                    }
                }
            }
        }
        CHECK(identical);
    }

    // batch broadcasting: one matrix reused across the batch, without materialising copies
    {
        const Tensor a = filled(Shape({2, 1, 2}), {1, 2, 3, 4});   // two [1,2] rows
        const Tensor shared = filled(Shape({1, 2, 2}), {1, 0, 0, 1});   // one identity, shared
        const Tensor c = matmul(a, shared);

        CHECK_EQ(c.shape(), Shape({2, 1, 2}));
        CHECK_EQ(c.at({0, 0, 0}), 1.0f);
        CHECK_EQ(c.at({0, 0, 1}), 2.0f);
        CHECK_EQ(c.at({1, 0, 0}), 3.0f);
        CHECK_EQ(c.at({1, 0, 1}), 4.0f);

        // a weight with no batch dimensions at all — the Linear case
        const Tensor activations = filled(Shape({1, 2, 2}), {1, 2, 3, 4});   // [B, T, D]
        const Tensor weight = filled(Shape({3, 2}), {1, 0, 0, 1, 1, 1});     // [F, D], HF layout
        const Tensor projected = matmul_nt(activations, weight);

        CHECK_EQ(projected.shape(), Shape({1, 2, 3}));   // [B, T, F]
        CHECK_EQ(projected.at({0, 0, 0}), 1.0f);         // [1,2] . [1,0]
        CHECK_EQ(projected.at({0, 0, 1}), 2.0f);         // [1,2] . [0,1]
        CHECK_EQ(projected.at({0, 0, 2}), 3.0f);         // [1,2] . [1,1]
        CHECK_EQ(projected.at({0, 1, 2}), 7.0f);         // [3,4] . [1,1]
    }

    // the shapes attention will actually use
    {
        const Tensor q{Shape({1, 16, 8, 64})};   // [B, H, T, Dh]
        const Tensor k{Shape({1, 16, 8, 64})};
        const Tensor scores = matmul_nt(q, k);   // Q * Kt, no transpose needed
        CHECK_EQ(scores.shape(), Shape({1, 16, 8, 8}));   // [B, H, T, T]

        const Tensor v{Shape({1, 16, 8, 64})};
        CHECK_EQ(matmul(scores, v).shape(), Shape({1, 16, 8, 64}));   // context, [B, H, T, Dh]

        // 16 independent matrix products per layer at B = 1
        CHECK_EQ(scores.shape()[0] * scores.shape()[1], size_t{16});
    }

    // matmul_nt batched agrees with matmul on the transposed slices
    {
        std::mt19937 engine(31415);
        const Tensor a = random_tensor(Shape({2, 3, 4}), engine);
        const Tensor b = random_tensor(Shape({2, 5, 4}), engine);   // [batch, N, K]
        const Tensor from_nt = matmul_nt(a, b);
        CHECK_EQ(from_nt.shape(), Shape({2, 3, 5}));

        bool identical = true;
        for (size_t e = 0; e < 2 && identical; ++e)
        {
            const Tensor a_slice = a.slice(0, e, 1).reshape(Shape({3, 4}));
            const Tensor b_slice = b.slice(0, e, 1).reshape(Shape({5, 4}));
            const Tensor expected = matmul(a_slice, b_slice.transpose(0, 1));

            for (size_t i = 0; i < 3 && identical; ++i)
            {
                for (size_t j = 0; j < 5; ++j)
                {
                    if (from_nt.at({e, i, j}) != expected.at({i, j}))
                    {
                        identical = false;
                        break;
                    }
                }
            }
        }
        CHECK(identical);
    }

    // deeper batch parts, and rank-2 against rank-4
    {
        CHECK_EQ(matmul(Tensor{Shape({2, 3, 4, 5})}, Tensor{Shape({2, 3, 5, 6})}).shape(),
                 Shape({2, 3, 4, 6}));
        CHECK_EQ(matmul(Tensor{Shape({2, 3, 4, 5})}, Tensor{Shape({5, 6})}).shape(),
                 Shape({2, 3, 4, 6}));
        CHECK_EQ(matmul(Tensor{Shape({4, 5})}, Tensor{Shape({2, 3, 5, 6})}).shape(),
                 Shape({2, 3, 4, 6}));
        CHECK_EQ(matmul(Tensor{Shape({1, 4, 5})}, Tensor{Shape({7, 5, 6})}).shape(),
                 Shape({7, 4, 6}));
    }

    // refusals
    {
        // batch 2 against batch 3, neither is 1
        CHECK_THROWS_AS(matmul(Tensor{Shape({2, 3, 4})}, Tensor{Shape({3, 4, 5})}),
                        std::invalid_argument);
        // the matrix dimensions still have to match
        CHECK_THROWS_AS(matmul(Tensor{Shape({2, 3, 4})}, Tensor{Shape({2, 5, 6})}),
                        std::invalid_argument);

        try
        {
            (void)matmul(Tensor{Shape({2, 3, 4})}, Tensor{Shape({3, 4, 5})});
        }
        catch (const std::invalid_argument& error)
        {
            const std::string message = error.what();
            CHECK(message.find("batch dimensions do not broadcast") != std::string::npos);
            CHECK(message.find("(2, 3, 4)") != std::string::npos);
            CHECK(message.find("(3, 4, 5)") != std::string::npos);
        }
    }

    // non-contiguous batched operands
    {
        // [2, 3, 4] transposed on its last two axes is [2, 4, 3], strided
        const Tensor a = filled(Shape({2, 2, 2}), {1, 2, 3, 4, 5, 6, 7, 8});
        const Tensor transposed = a.transpose(1, 2);
        CHECK(!transposed.is_contiguous());

        const Tensor c = matmul(transposed, filled(Shape({2, 2}), {1, 0, 0, 1}));
        CHECK_EQ(c.shape(), Shape({2, 2, 2}));
        CHECK_EQ(c.at({0, 0, 0}), 1.0f);   // A[0] transposed is [[1,3],[2,4]]
        CHECK_EQ(c.at({0, 0, 1}), 3.0f);
        CHECK_EQ(c.at({0, 1, 0}), 2.0f);
        CHECK_EQ(c.at({1, 1, 1}), 8.0f);

        // a batch built by slicing, with a non-zero offset
        const Tensor stack = filled(Shape({3, 2, 2}), {9, 9, 9, 9, 1, 2, 3, 4, 5, 6, 7, 8});
        const Tensor tail = stack.slice(0, 1, 2);   // drops the first matrix, offset 4
        CHECK_EQ(tail.offset(), size_t{4});
        const Tensor product = matmul(tail, filled(Shape({2, 2}), {1, 0, 0, 1}));
        CHECK_EQ(product.at({0, 0, 0}), 1.0f);
        CHECK_EQ(product.at({1, 1, 1}), 8.0f);
    }

    // edge cases
    {
        // an empty batch dimension
        CHECK_EQ(matmul(Tensor{Shape({0, 2, 3})}, Tensor{Shape({0, 3, 4})}).shape(),
                 Shape({0, 2, 4}));
        CHECK_EQ(matmul(Tensor{Shape({0, 2, 3})}, Tensor{Shape({3, 4})}).numel(), size_t{0});

        // K == 0 inside a batch
        const Tensor zeros = matmul(Tensor{Shape({2, 3, 0})}, Tensor{Shape({2, 0, 4})});
        CHECK_EQ(zeros.shape(), Shape({2, 3, 4}));
        CHECK_EQ(zeros.at({1, 2, 3}), 0.0f);

        // a batch of one behaves like the rank-2 case, one axis deeper
        const Tensor single = matmul(filled(Shape({1, 2, 2}), {1, 2, 3, 4}),
                                     filled(Shape({1, 2, 2}), {5, 6, 7, 8}));
        CHECK_EQ(single.at({0, 0, 0}), 19.0f);
        CHECK_EQ(single.at({0, 1, 1}), 50.0f);
    }

    return VEDA_TEST_SUMMARY("BatchedMatmulTest");
}
