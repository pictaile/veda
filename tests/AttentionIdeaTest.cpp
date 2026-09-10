// E7.S1.T1 — the idea (a Learn task; the mechanism is built across S2–S5)
//
// The whole of attention, on three tokens of two dimensions, assembled from the ops E2 already
// provides. Every number here is checkable by hand.

#include "Matmul.h"
#include "Shape.h"
#include "Softmax.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::ops::matmul;
using veda::ops::matmul_nt;
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
} // namespace

int main()
{
    // Q = K = [[1,0],[0,1],[1,1]],  V = [[10,0],[0,10],[5,5]]
    const Tensor q = filled(Shape({3, 2}), {1, 0, 0, 1, 1, 1});
    const Tensor k = filled(Shape({3, 2}), {1, 0, 0, 1, 1, 1});
    const Tensor v = filled(Shape({3, 2}), {10, 0, 0, 10, 5, 5});

    // --- the dot product as similarity -----------------------------------------------------------
    {
        const Tensor scores = matmul_nt(q, k);
        CHECK_EQ(scores.shape(), Shape({3, 3}));

        // every entry is q[t] . k[s], and every one is checkable by eye
        CHECK_EQ(scores.at({0, 0}), 1.0f);   // [1,0].[1,0]
        CHECK_EQ(scores.at({0, 1}), 0.0f);   // [1,0].[0,1] — perpendicular, no relationship
        CHECK_EQ(scores.at({0, 2}), 1.0f);
        CHECK_EQ(scores.at({1, 0}), 0.0f);
        CHECK_EQ(scores.at({1, 1}), 1.0f);
        CHECK_EQ(scores.at({2, 0}), 1.0f);
        CHECK_EQ(scores.at({2, 1}), 1.0f);
        CHECK_EQ(scores.at({2, 2}), 2.0f);   // [1,1].[1,1] — itself, and it scores highest

        // a position attends to itself, and here it does so most strongly
        CHECK(scores.at({2, 2}) > scores.at({2, 0}));
        CHECK(scores.at({2, 2}) > scores.at({2, 1}));

        // magnitude matters, not just direction: a longer vector scores higher along the same line
        const Tensor longer = filled(Shape({1, 2}), {2, 2});
        CHECK_EQ(matmul_nt(longer, k).at({0, 2}), 2.0f * scores.at({2, 2}));
    }

    // --- the score matrix is generally NOT symmetric ---------------------------------------------
    // It is symmetric above only because Q and K are equal. In the model they are different
    // projections, and how much t cares about s is not how much s cares about t.
    {
        const Tensor other_k = filled(Shape({3, 2}), {1, 0, 2, 0, 0, 3});
        const Tensor scores = matmul_nt(q, other_k);
        CHECK(scores.at({0, 1}) != scores.at({1, 0}));
        CHECK_EQ(scores.at({0, 1}), 2.0f);
        CHECK_EQ(scores.at({1, 0}), 0.0f);
    }

    // --- nothing is selected: everything contributes ----------------------------------------------
    {
        const Tensor scores = matmul_nt(q, k);
        const Tensor weights = softmax(scores, 1);

        // each row is a distribution over the whole sequence
        for (size_t t = 0; t < 3; ++t)
        {
            float sum = 0.0f;
            for (size_t s = 0; s < 3; ++s)
            {
                CHECK(weights.at({t, s}) > 0.0f);   // every position contributes something
                sum += weights.at({t, s});
            }
            CHECK_NEAR(sum, 1.0, 1e-6);
        }

        // position 2: weights about [0.21, 0.21, 0.58]
        CHECK_NEAR(weights.at({2, 0}), 0.2119, 1e-3);
        CHECK_NEAR(weights.at({2, 1}), 0.2119, 1e-3);
        CHECK_NEAR(weights.at({2, 2}), 0.5761, 1e-3);

        // and the output is a blend of all three values, not a lookup of one
        const Tensor out = matmul(weights, v);
        CHECK_EQ(out.shape(), Shape({3, 2}));
        CHECK_NEAR(out.at({2, 0}), 5.0, 1e-3);
        CHECK_NEAR(out.at({2, 1}), 5.0, 1e-3);

        // position 0 leans towards value 0 but still carries the others
        CHECK(out.at({0, 0}) > out.at({0, 1}));
        CHECK(out.at({0, 1}) > 0.0f);
    }

    // --- the whole mechanism in one expression ----------------------------------------------------
    // Attention(Q, K, V) = softmax(QKt) . V     — scaling and masking are S3 and S4
    {
        const Tensor out = matmul(softmax(matmul_nt(q, k), 1), v);
        CHECK_EQ(out.shape(), Shape({3, 2}));
        CHECK_NEAR(out.at({2, 0}), 5.0, 1e-3);

        // every output row is a convex combination of the value rows, so it lies inside their range
        for (size_t t = 0; t < 3; ++t)
        {
            for (size_t d = 0; d < 2; ++d)
            {
                CHECK(out.at({t, d}) >= 0.0f);
                CHECK(out.at({t, d}) <= 10.0f);
            }
        }
    }

    // --- the quadratic, in numbers ----------------------------------------------------------------
    {
        const size_t H = 16;
        const size_t Dh = 128;

        // the score matrix at a short prompt and at a long context
        CHECK_EQ(H * 8 * 8, size_t{1024});
        CHECK_EQ(H * 1024 * 1024, size_t{16'777'216});                 // 16.8M floats per layer
        CHECK_EQ(H * 1024 * 1024 * sizeof(float), size_t{67'108'864}); // 64 MB, per layer

        // it grows with the square: 4x the context is 16x the matrix
        CHECK_EQ((H * 2048 * 2048) / (H * 1024 * 1024), size_t{4});
        CHECK_EQ((H * 4096 * 4096) / (H * 1024 * 1024), size_t{16});

        // arithmetic: B*H*T*S*Dh multiply-adds — negligible at T=8, dominant at T=1024
        const size_t at_short = H * 8 * 8 * Dh;
        const size_t at_long = H * 1024 * 1024 * Dh;
        CHECK(at_long / at_short > 16000);
    }

    return VEDA_TEST_SUMMARY("AttentionIdeaTest");
}
