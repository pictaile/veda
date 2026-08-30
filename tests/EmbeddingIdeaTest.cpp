// E6.S1.T1 — a token id is a row index (a Learn task; nn::Embedding arrives in T2)
//
// The claim worth checking in code: the textbook one-hot matmul and a row copy are the same
// function, and only one of them is an implementation.

#include "Matmul.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::ops::matmul;

namespace
{
// A four-token vocabulary in three dimensions.
Tensor table()
{
    Tensor t{Shape({4, 3})};
    const std::vector<float> rows = {0.1f, 0.2f, 0.3f, 1.0f, 1.1f, 1.2f,
                                     2.0f, 2.1f, 2.2f, 3.0f, 3.1f, 3.2f};
    for (size_t i = 0; i < rows.size(); ++i)
    {
        t.data()[i] = rows[i];
    }
    return t;
}

Tensor one_hot(size_t id, size_t vocabulary)
{
    Tensor t{Shape({1, vocabulary})};
    t.data()[id] = 1.0f;
    return t;
}
} // namespace

int main()
{
    const Tensor w = table();

    // the one-hot route: mathematically the definition, and computationally absurd
    {
        const Tensor selected = matmul(one_hot(2, 4), w);
        CHECK_EQ(selected.shape(), Shape({1, 3}));
        CHECK_EQ(selected.at({0, 0}), 2.0f);
        CHECK_EQ(selected.at({0, 1}), 2.1f);
        CHECK_EQ(selected.at({0, 2}), 2.2f);

        // the lookup route: the same three numbers, copied
        CHECK_EQ(selected.at({0, 0}), w.at({2, 0}));
        CHECK_EQ(selected.at({0, 1}), w.at({2, 1}));
        CHECK_EQ(selected.at({0, 2}), w.at({2, 2}));

        // and it holds for every id, not just this one
        for (size_t id = 0; id < 4; ++id)
        {
            const Tensor row = matmul(one_hot(id, 4), w);
            for (size_t d = 0; d < 3; ++d)
            {
                CHECK_EQ(row.at({0, d}), w.at({id, d}));
            }
        }
    }

    // what the equivalence costs, at the real scale
    {
        const size_t V = 151936;
        const size_t D = 1024;

        const size_t one_hot_multiplications = V * D;   // and all but D of them are by zero
        const size_t lookup_copies = D;
        CHECK_EQ(one_hot_multiplications, size_t{155'582'464});
        CHECK_EQ(one_hot_multiplications / lookup_copies, V);

        // the table itself: 311 MB as bf16 on disk, twice that resident (AD1)
        CHECK_EQ(V * D * 2, size_t{311'164'928});
        CHECK_EQ(V * D * 4, size_t{622'329'856});
    }

    // a row means nothing alone — only relative to other rows. Two rows of the same table can be
    // compared; a row and a raw number cannot.
    {
        // rows 2 and 3 differ by a constant offset here, which is what "similar direction" means
        float distance_23 = 0.0f;
        float distance_03 = 0.0f;
        for (size_t d = 0; d < 3; ++d)
        {
            const float near = w.at({2, d}) - w.at({3, d});
            const float far = w.at({0, d}) - w.at({3, d});
            distance_23 += near * near;
            distance_03 += far * far;
        }
        CHECK(distance_23 < distance_03);
    }

    // the shape story: [B, T] ids become [B, T, D], and nothing mixes between positions
    {
        const size_t B = 1;
        const size_t T = 8;
        const size_t D = 1024;
        CHECK_EQ(B * T * D, size_t{8192});          // 32 KB of activations for eight tokens
        CHECK_EQ(B * T * D * sizeof(float), size_t{32768});
    }

    // tied embeddings: the same table also serves as the LM head, so a forward pass reads it twice
    {
        // final hidden [1, D] against the table [V, D] via matmul_nt -> [1, V] logits
        Tensor hidden{Shape({1, 3})};
        hidden.data()[0] = 2.0f;
        hidden.data()[1] = 2.1f;
        hidden.data()[2] = 2.2f;

        const Tensor logits = veda::ops::matmul_nt(hidden, w);
        CHECK_EQ(logits.shape(), Shape({1, 4}));

        // the vector points most closely at the row it was copied from — which is how a token is
        // chosen at the end of the model
        size_t best = 0;
        for (size_t v = 1; v < 4; ++v)
        {
            best = logits.at({0, v}) > logits.at({0, best}) ? v : best;
        }
        CHECK_EQ(best, size_t{3});   // the largest row wins on raw dot product, before any norm
        CHECK(logits.at({0, 2}) > logits.at({0, 0}));
    }

    return VEDA_TEST_SUMMARY("EmbeddingIdeaTest");
}
