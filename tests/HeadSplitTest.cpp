// E7.S2.T2 (the head split) and T3 (the score matrix)

#include "Attention.h"
#include "Linear.h"
#include "Matmul.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <stdexcept>
#include <string>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::model::attention_scores;
using veda::model::split_heads;
using veda::nn::Linear;

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

Tensor counting(const Shape& shape)
{
    Tensor t{shape};
    for (size_t i = 0; i < shape.size(); ++i)
    {
        t.data()[i] = static_cast<float>(i);
    }
    return t;
}
} // namespace

int main()
{
    // --- T2: the worked example ------------------------------------------------------------------
    // [1, 2, 4] with H=2, Dh=2:
    //   token 0: [a0 a1 b0 b1]   token 1: [c0 c1 d0 d1]
    //   head 0: [[a0 a1], [c0 c1]]      head 1: [[b0 b1], [d0 d1]]
    {
        const Tensor projected = counting(Shape({1, 2, 4}));   // 0..7
        const Tensor heads = split_heads(projected, 2, 2);

        CHECK_EQ(heads.shape(), Shape({1, 2, 2, 2}));

        CHECK_EQ(heads.at({0, 0, 0, 0}), 0.0f);   // head 0, token 0
        CHECK_EQ(heads.at({0, 0, 0, 1}), 1.0f);
        CHECK_EQ(heads.at({0, 0, 1, 0}), 4.0f);   // head 0, token 1
        CHECK_EQ(heads.at({0, 0, 1, 1}), 5.0f);
        CHECK_EQ(heads.at({0, 1, 0, 0}), 2.0f);   // head 1, token 0
        CHECK_EQ(heads.at({0, 1, 1, 1}), 7.0f);

        // head 0 takes the FIRST Dh of each token, not the first half of the tensor
        CHECK_EQ(heads.at({0, 0, 1, 0}), projected.at({0, 1, 0}));
        CHECK_EQ(heads.at({0, 1, 1, 0}), projected.at({0, 1, 2}));
    }

    // the property, for every index: out[b,h,t,d] == in[b,t,h*Dh+d]
    {
        const size_t B = 2;
        const size_t T = 3;
        const size_t H = 4;
        const size_t DH = 5;
        const Tensor projected = counting(Shape({B, T, H * DH}));
        const Tensor heads = split_heads(projected, H, DH);

        bool matches = true;
        for (size_t b = 0; b < B && matches; ++b)
        {
            for (size_t h = 0; h < H && matches; ++h)
            {
                for (size_t t = 0; t < T && matches; ++t)
                {
                    for (size_t d = 0; d < DH; ++d)
                    {
                        if (heads.at({b, h, t, d}) != projected.at({b, t, h * DH + d}))
                        {
                            matches = false;
                            break;
                        }
                    }
                }
            }
        }
        CHECK(matches);
    }

    // *** not one float moves ***
    {
        const Tensor projected = counting(Shape({1, 8, 2048}));
        const Tensor heads = split_heads(projected, 16, 128);

        CHECK_EQ(heads.shape(), Shape({1, 16, 8, 128}));
        CHECK(heads.data() == projected.data());              // the same pointer
        CHECK(heads.storage() == projected.storage());        // the same buffer
        CHECK_EQ(heads.numel(), projected.numel());
        CHECK(!heads.is_contiguous());                        // and it is a strided view

        // the strides tell the story: stepping a head costs Dh, stepping a token costs H*Dh
        CHECK_EQ(heads.strides(), std::vector<size_t>({16384, 128, 2048, 1}));
    }

    // the GQA pair: Q and K are not the same shape, and the same function serves both
    {
        const Tensor q = Tensor{Shape({1, 8, 2048})};      // H   = 16
        const Tensor k = Tensor{Shape({1, 8, 1024})};      // Hkv = 8
        CHECK_EQ(split_heads(q, 16, 128).shape(), Shape({1, 16, 8, 128}));
        CHECK_EQ(split_heads(k, 8, 128).shape(), Shape({1, 8, 8, 128}));

        // H = 1, the shape the epic milestone uses
        CHECK_EQ(split_heads(Tensor{Shape({1, 3, 2})}, 1, 2).shape(), Shape({1, 1, 3, 2}));
    }

    // the projections that feed it, assembled from nn::Linear
    {
        const size_t D = 4;
        const size_t H = 2;
        const size_t DH = 3;
        const Tensor x{Shape({1, 5, D})};                       // [B, T, D]
        const Linear q_proj(Tensor{Shape({H * DH, D})});        // [H*Dh, D], the file's layout

        const Tensor projected = q_proj.forward(x);
        CHECK_EQ(projected.shape(), Shape({1, 5, H * DH}));
        CHECK_EQ(split_heads(projected, H, DH).shape(), Shape({1, H, 5, DH}));
    }

    // refusals
    {
        CHECK_THROWS_AS(split_heads(Tensor{Shape({1, 2, 5})}, 2, 2), std::invalid_argument);
        CHECK_THROWS_AS(split_heads(Tensor{Shape({2, 4})}, 2, 2), std::invalid_argument);
        CHECK_THROWS_AS(split_heads(Tensor{Shape({1, 2, 2, 2})}, 2, 2), std::invalid_argument);
        CHECK_THROWS_AS(split_heads(Tensor{Shape({1, 2, 4})}, 0, 2), std::invalid_argument);
        CHECK_THROWS_AS(split_heads(Tensor{Shape({1, 2, 4})}, 2, 0), std::invalid_argument);

        try
        {
            (void)split_heads(Tensor{Shape({1, 2, 5})}, 2, 2);
        }
        catch (const std::invalid_argument& error)
        {
            const std::string message = error.what();
            CHECK(message.find("5") != std::string::npos);
            CHECK(message.find("2 heads of 2") != std::string::npos);
        }
    }

    // edge cases
    {
        CHECK_EQ(split_heads(Tensor{Shape({1, 0, 4})}, 2, 2).shape(), Shape({1, 2, 0, 2}));
        CHECK_EQ(split_heads(Tensor{Shape({1, 3, 4})}, 4, 1).shape(), Shape({1, 4, 3, 1}));
    }

    // --- T3: the score matrix, hand-verified -----------------------------------------------------
    // Q = K = [[1,0],[0,1],[1,1]] as [1, 1, 3, 2]
    {
        const Tensor q = filled(Shape({1, 1, 3, 2}), {1, 0, 0, 1, 1, 1});
        const Tensor k = filled(Shape({1, 1, 3, 2}), {1, 0, 0, 1, 1, 1});
        const Tensor scores = attention_scores(q, k);

        CHECK_EQ(scores.shape(), Shape({1, 1, 3, 3}));
        CHECK_EQ(scores.at({0, 0, 0, 0}), 1.0f);
        CHECK_EQ(scores.at({0, 0, 0, 1}), 0.0f);
        CHECK_EQ(scores.at({0, 0, 0, 2}), 1.0f);
        CHECK_EQ(scores.at({0, 0, 1, 0}), 0.0f);
        CHECK_EQ(scores.at({0, 0, 1, 1}), 1.0f);
        CHECK_EQ(scores.at({0, 0, 1, 2}), 1.0f);
        CHECK_EQ(scores.at({0, 0, 2, 0}), 1.0f);
        CHECK_EQ(scores.at({0, 0, 2, 1}), 1.0f);
        CHECK_EQ(scores.at({0, 0, 2, 2}), 2.0f);

        CHECK(scores.is_contiguous());
        CHECK(scores.storage() != q.storage());
        CHECK_EQ(q.at({0, 0, 2, 0}), 1.0f);   // inputs untouched
    }

    // straight from split_heads: non-contiguous operands, and per-head batching at H > 1
    {
        // two heads of two dimensions, three tokens
        //   token t, head h occupies projected[0, t, 2h .. 2h+1]
        const Tensor projected = filled(Shape({1, 3, 4}), {1, 0, 2, 0,     // t0: h0 [1,0] h1 [2,0]
                                                           0, 1, 0, 3,     // t1: h0 [0,1] h1 [0,3]
                                                           1, 1, 1, 1});   // t2: h0 [1,1] h1 [1,1]
        const Tensor heads = split_heads(projected, 2, 2);
        CHECK(!heads.is_contiguous());

        const Tensor scores = attention_scores(heads, heads);
        CHECK_EQ(scores.shape(), Shape({1, 2, 3, 3}));

        // head 0: Q = K = [[1,0],[0,1],[1,1]] — the matrix above
        CHECK_EQ(scores.at({0, 0, 2, 2}), 2.0f);
        CHECK_EQ(scores.at({0, 0, 0, 1}), 0.0f);

        // head 1: Q = K = [[2,0],[0,3],[1,1]] — a different matrix from the same buffer
        CHECK_EQ(scores.at({0, 1, 0, 0}), 4.0f);   // [2,0].[2,0]
        CHECK_EQ(scores.at({0, 1, 0, 1}), 0.0f);   // [2,0].[0,3]
        CHECK_EQ(scores.at({0, 1, 0, 2}), 2.0f);   // [2,0].[1,1]
        CHECK_EQ(scores.at({0, 1, 1, 1}), 9.0f);   // [0,3].[0,3]
        CHECK_EQ(scores.at({0, 1, 2, 2}), 2.0f);

        // the two heads are independent: same buffer, different score matrices
        CHECK_EQ(scores.at({0, 0, 0, 0}), 1.0f);
        CHECK(scores.at({0, 0, 0, 0}) != scores.at({0, 1, 0, 0}));
        CHECK(scores.at({0, 0, 1, 1}) != scores.at({0, 1, 1, 1}));
    }

    // the score matrix is generally not symmetric
    {
        const Tensor q = filled(Shape({1, 1, 2, 2}), {1, 0, 0, 1});
        const Tensor k = filled(Shape({1, 1, 2, 2}), {1, 0, 2, 0});
        const Tensor scores = attention_scores(q, k);
        CHECK_EQ(scores.at({0, 0, 0, 1}), 2.0f);
        CHECK_EQ(scores.at({0, 0, 1, 0}), 0.0f);
        CHECK(scores.at({0, 0, 0, 1}) != scores.at({0, 0, 1, 0}));
    }

    // the shapes the model uses, including the cached-generation one
    {
        CHECK_EQ(attention_scores(Tensor{Shape({1, 16, 8, 128})}, Tensor{Shape({1, 16, 8, 128})})
                     .shape(),
                 Shape({1, 16, 8, 8}));
        CHECK_EQ(attention_scores(Tensor{Shape({1, 16, 1, 128})}, Tensor{Shape({1, 16, 40, 128})})
                     .shape(),
                 Shape({1, 16, 1, 40}));   // one new query, 40 cached keys (E12)
    }

    // it agrees with the long way round: matmul(Q, K transposed on its last two axes)
    {
        const Tensor q = counting(Shape({1, 2, 3, 4}));
        const Tensor k = counting(Shape({1, 2, 5, 4}));
        const Tensor scores = attention_scores(q, k);
        const Tensor by_hand = veda::ops::matmul(q, k.transpose(2, 3));

        CHECK_EQ(scores.shape(), by_hand.shape());
        bool identical = true;
        for (size_t h = 0; h < 2 && identical; ++h)
        {
            for (size_t t = 0; t < 3 && identical; ++t)
            {
                for (size_t s = 0; s < 5; ++s)
                {
                    if (scores.at({0, h, t, s}) != by_hand.at({0, h, t, s}))
                    {
                        identical = false;
                        break;
                    }
                }
            }
        }
        CHECK(identical);
    }

    // refusals and degenerate shapes
    {
        CHECK_THROWS_AS(attention_scores(Tensor{Shape({1, 1, 3, 2})}, Tensor{Shape({1, 1, 3, 4})}),
                        std::invalid_argument);
        CHECK_THROWS_AS(attention_scores(Tensor{Shape({3, 2})}, Tensor{Shape({3, 2})}),
                        std::invalid_argument);
        try
        {
            (void)attention_scores(Tensor{Shape({1, 1, 3, 2})}, Tensor{Shape({1, 1, 3, 4})});
        }
        catch (const std::invalid_argument& error)
        {
            CHECK(std::string(error.what()).find("head dimensions disagree") != std::string::npos);
        }

        CHECK_EQ(attention_scores(Tensor{Shape({1, 1, 1, 3})}, Tensor{Shape({1, 1, 1, 3})}).shape(),
                 Shape({1, 1, 1, 1}));
        CHECK_EQ(attention_scores(Tensor{Shape({1, 1, 0, 3})}, Tensor{Shape({1, 1, 4, 3})}).numel(),
                 size_t{0});
    }

    return VEDA_TEST_SUMMARY("HeadSplitTest");
}
