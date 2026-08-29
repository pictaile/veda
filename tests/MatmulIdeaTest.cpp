// E2.S2.T3 — What matmul computes, and why it dominates the cost (a Learn task; T4 implements it)

#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <optional>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;

namespace
{
// [M, K] x [K, N] -> [M, N]: the inner dimensions must match and they vanish from the result.
std::optional<Shape> matmul_shape(const Shape& a, const Shape& b)
{
    if (a.rank() != 2 || b.rank() != 2 || a[1] != b[0])
    {
        return std::nullopt;
    }
    return Shape({a[0], b[1]});
}

// C[i, j] = sum over k of A[i, k] * B[k, j] — one row against one column.
float dot_of_row_and_column(const Tensor& a, const Tensor& b, size_t i, size_t j)
{
    float sum = 0.0f;
    for (size_t k = 0; k < a.shape()[1]; ++k)
    {
        sum += a.at({i, k}) * b.at({k, j});
    }
    return sum;
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
    // the shape rule: inner dimensions match, outer ones survive
    CHECK_EQ(*matmul_shape(Shape({2, 3}), Shape({3, 4})), Shape({2, 4}));
    CHECK_EQ(*matmul_shape(Shape({1, 3}), Shape({3, 1})), Shape({1, 1}));
    CHECK_EQ(*matmul_shape(Shape({8, 1024}), Shape({1024, 3072})), Shape({8, 3072}));
    CHECK(!matmul_shape(Shape({3, 4}), Shape({2, 3})).has_value());   // not commutative
    CHECK(!matmul_shape(Shape({2, 3}), Shape({2, 3})).has_value());

    // the 2x2 every later test starts from, worked out one dot product at a time
    {
        const Tensor a = filled(Shape({2, 2}), {1, 2, 3, 4});
        const Tensor b = filled(Shape({2, 2}), {5, 6, 7, 8});

        CHECK_EQ(dot_of_row_and_column(a, b, 0, 0), 19.0f);   // 1*5 + 2*7
        CHECK_EQ(dot_of_row_and_column(a, b, 0, 1), 22.0f);   // 1*6 + 2*8
        CHECK_EQ(dot_of_row_and_column(a, b, 1, 0), 43.0f);   // 3*5 + 4*7
        CHECK_EQ(dot_of_row_and_column(a, b, 1, 1), 50.0f);   // 3*6 + 4*8

        // A*B and B*A are different computations, not just a different order
        CHECK(dot_of_row_and_column(b, a, 0, 0) != dot_of_row_and_column(a, b, 0, 0));
    }

    // a row times a column collapses to a single number
    {
        const Tensor row = filled(Shape({1, 3}), {1, 2, 3});
        const Tensor column = filled(Shape({3, 1}), {1, 0, 2});
        CHECK_EQ(*matmul_shape(row.shape(), column.shape()), Shape({1, 1}));
        CHECK_EQ(dot_of_row_and_column(row, column, 0, 0), 7.0f);   // 1 + 0 + 6
    }

    // the cost: M*N*K multiply-adds for M*N outputs — the only op in Veda that is not linear
    {
        const size_t D = 1024;      // hidden size
        const size_t F = 3072;      // feed-forward inner size
        const size_t KV = 512;      // Hkv * Dh = 8 * 64
        const size_t V = 151669;    // vocabulary
        const size_t layers = 28;

        const size_t attention = D * D + D * KV + D * KV + D * D;   // q, k, v, o
        const size_t ffn = D * F + D * F + F * D;                   // gate, up, down
        const size_t per_layer = attention + ffn;
        CHECK_EQ(per_layer, size_t{12'582'912});                    // ~12.6M MACs per token

        const size_t all_layers = per_layer * layers;
        const size_t lm_head = D * V;
        CHECK_EQ(lm_head, size_t{155'309'056});

        // the LM head alone is roughly a third of a token's work — which is why generation
        // computes it for the last position only (the slice from E1.S2.T7)
        CHECK(lm_head * 3 > all_layers);
        CHECK(lm_head * 2 < all_layers);

        // ~0.5G MACs, i.e. about 1 GFLOP per token, before the prefix is reprocessed
        CHECK(all_layers + lm_head > 500'000'000);
        CHECK(all_layers + lm_head < 550'000'000);

        // doubling the hidden size quadruples a projection: both M and K grow with it
        CHECK_EQ((2 * D) * (2 * D), 4 * D * D);
    }

    // attention's own products are quadratic in T, not in D: small at short prompts, dominant later
    {
        const size_t Dh = 64;
        for (const size_t T : {8u, 1024u})
        {
            const size_t scores = T * T * Dh;    // [T,Dh] x [Dh,T]
            const size_t context = T * Dh * T;   // [T,T] x [T,Dh]
            CHECK_EQ(scores, context);
            CHECK_EQ(scores, T * T * Dh);
        }
        CHECK(1024u * 1024u * 64u > 128u * (8u * 8u * 64u));   // growth is quadratic in T
    }

    return VEDA_TEST_SUMMARY("MatmulIdeaTest");
}
