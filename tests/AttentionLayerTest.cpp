// E7.S5.T7 — ScaledDotProductAttention, and the epic milestone

#include "Attention.h"
#include "Linear.h"
#include "ScaledDotProductAttention.h"
#include "Shape.h"
#include "Softmax.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::model::ScaledDotProductAttention;
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
    // *** THE MILESTONE: a full pass at T = 3, H = 1, Dh = 2, every number computed by hand ***
    //
    //   x = [[1,0],[0,1],[1,1]],  q_proj = k_proj = o_proj = I,  v_proj = 10*I
    //   scores  = [[1,0,1],[0,1,1],[1,1,2]]                        (S2)
    //   scaled  = / sqrt(2)                                        (S3)
    //   masked  = -inf above the diagonal                          (S4)
    //   weights = [[1,0,0], [0.330,0.670,0], [0.248,0.248,0.503]]  (S4)
    //   context = [[10,0], [3.30,6.70], [7.52,7.52]]               (S5)
    {
        ScaledDotProductAttention attention(
            {Linear(identity(2)), Linear(identity(2)),
             Linear(filled(Shape({2, 2}), {10, 0, 0, 10})), Linear(identity(2))},
            {1, 1, 2, std::nullopt});

        const Tensor x = filled(Shape({1, 3, 2}), {1, 0, 0, 1, 1, 1});
        const Tensor out = attention.forward(x);

        CHECK_EQ(out.shape(), Shape({1, 3, 2}));

        // position 0 attends only to itself, so it gets value row 0 exactly — the causal
        // guarantee, end to end
        CHECK_NEAR(out.at({0, 0, 0}), 10.0, 1e-4);
        CHECK_NEAR(out.at({0, 0, 1}), 0.0, 1e-6);

        // position 1: weights [0.3302, 0.6698] over values [10,0] and [0,10]
        CHECK_NEAR(out.at({0, 1, 0}), 3.302, 1e-3);
        CHECK_NEAR(out.at({0, 1, 1}), 6.698, 1e-3);

        // position 2: weights [0.2483, 0.2483, 0.5035] over [10,0], [0,10], [10,10].
        // Note these are the SCALED weights — the unscaled ones from S1 were [0.212, 0.212, 0.576].
        CHECK_NEAR(out.at({0, 2, 0}), 7.517, 1e-3);
        CHECK_NEAR(out.at({0, 2, 1}), 7.517, 1e-3);

        CHECK(out.is_contiguous());
        CHECK_EQ(x.at({0, 0, 0}), 1.0f);   // the input is unmodified
    }

    // the same pass, step by step, against the architecture's shape table
    {
        const size_t D = 2;
        const size_t H = 1;
        const size_t DH = 2;

        const Tensor x = filled(Shape({1, 3, D}), {1, 0, 0, 1, 1, 1});
        const Linear q_proj(identity(D));

        const Tensor q = veda::model::split_heads(q_proj.forward(x), H, DH);
        CHECK_EQ(q.shape(), Shape({1, 1, 3, 2}));            // [B, H, T, Dh]

        const Tensor scores = veda::model::attention_scores(q, q);
        CHECK_EQ(scores.shape(), Shape({1, 1, 3, 3}));       // [B, H, T, T]
        CHECK_EQ(scores.at({0, 0, 2, 2}), 2.0f);

        const Tensor scaled = veda::model::scale_scores(scores, DH);
        CHECK_NEAR(scaled.at({0, 0, 2, 2}), 1.41421, 1e-4);

        const Tensor weights = veda::ops::softmax(veda::model::apply_causal_mask(scaled), 3);
        CHECK_EQ(weights.at({0, 0, 0, 1}), 0.0f);            // exactly zero, from S4
        CHECK_NEAR(weights.at({0, 0, 1, 0}), 0.3302, 1e-3);
        CHECK_NEAR(weights.at({0, 0, 2, 2}), 0.5035, 1e-3);

        const Tensor values = filled(Shape({1, 1, 3, 2}), {10, 0, 0, 10, 10, 10});
        const Tensor context = veda::model::attention_context(weights, values);
        CHECK_EQ(context.shape(), Shape({1, 1, 3, 2}));      // [B, H, T, Dh]

        const Tensor merged = veda::model::merge_heads(context);
        CHECK_EQ(merged.shape(), Shape({1, 3, 2}));          // [B, T, H*Dh]
        CHECK_NEAR(merged.at({0, 2, 0}), 7.517, 1e-3);
    }

    // H = 2, with the two heads seeing different projections — a head-indexing bug survives H = 1
    {
        const size_t D = 4;
        const size_t H = 2;
        const size_t DH = 2;

        // q and k are the identity, so each head compares its own two channels
        ScaledDotProductAttention attention(
            {Linear(identity(D)), Linear(identity(D)),
             Linear(filled(Shape({4, 4}), {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5})),
             Linear(identity(D))},
            {H, H, DH, std::nullopt});

        CHECK_EQ(attention.heads(), size_t{2});
        CHECK_EQ(attention.head_dim(), size_t{2});

        //           head 0 channels | head 1 channels
        const Tensor x = filled(Shape({1, 2, D}), {1, 0, /**/ 0, 1,
                                                   0, 1, /**/ 0, 1});
        const Tensor out = attention.forward(x);
        CHECK_EQ(out.shape(), Shape({1, 2, D}));

        // head 0: token 1's query [0,1] is orthogonal to token 0's key [1,0], so the weights are
        // softmax([0, 1/sqrt(2)]) = [0.330, 0.670]
        CHECK_NEAR(out.at({0, 1, 0}), 0.330, 1e-3);
        CHECK_NEAR(out.at({0, 1, 1}), 0.670, 1e-3);

        // head 1: both tokens have the same key [0,1], so the weights are [0.5, 0.5] and the two
        // identical values average to themselves — a different distribution from head 0
        CHECK_NEAR(out.at({0, 1, 3}), 5.0, 1e-3);
        CHECK(out.at({0, 1, 1}) != out.at({0, 1, 3}));

        // token 0 sees only itself in both heads
        CHECK_NEAR(out.at({0, 0, 0}), 1.0, 1e-5);
        CHECK_NEAR(out.at({0, 0, 3}), 5.0, 1e-5);
    }

    // *** attention has no notion of position — which is exactly what E8 fixes ***
    // Reversing the sequence and running a non-causal comparison would show it directly; with the
    // causal mask the cleanest statement is that the FIRST token's output depends only on itself,
    // whatever follows it.
    {
        ScaledDotProductAttention attention(
            {Linear(identity(2)), Linear(identity(2)),
             Linear(filled(Shape({2, 2}), {10, 0, 0, 10})), Linear(identity(2))},
            {1, 1, 2, std::nullopt});

        const Tensor a = filled(Shape({1, 3, 2}), {1, 0, 0, 1, 1, 1});
        const Tensor b = filled(Shape({1, 3, 2}), {1, 0, 5, 5, 9, 2});   // same first token
        const Tensor out_a = attention.forward(a);
        const Tensor out_b = attention.forward(b);

        CHECK_EQ(out_a.at({0, 0, 0}), out_b.at({0, 0, 0}));
        CHECK_EQ(out_a.at({0, 0, 1}), out_b.at({0, 0, 1}));

        // and two identical tokens at different positions produce identical outputs when the
        // prefixes match — the layer cannot tell them apart by position
        const Tensor twice = filled(Shape({1, 2, 2}), {3, 4, 3, 4});
        const Tensor out_twice = attention.forward(twice);
        CHECK_NEAR(out_twice.at({0, 0, 0}), out_twice.at({0, 1, 0}), 1e-4);
        CHECK_NEAR(out_twice.at({0, 0, 1}), out_twice.at({0, 1, 1}), 1e-4);
    }

    // batch and single-position cases
    {
        ScaledDotProductAttention attention(
            {Linear(identity(2)), Linear(identity(2)), Linear(identity(2)), Linear(identity(2))},
            {1, 1, 2, std::nullopt});

        CHECK_EQ(attention.forward(Tensor{Shape({2, 3, 2})}).shape(), Shape({2, 3, 2}));
        CHECK_EQ(attention.forward(Tensor{Shape({1, 1, 2})}).shape(), Shape({1, 1, 2}));

        // T = 1: the only weight is 1, so the output is the value of that token
        const Tensor single = filled(Shape({1, 1, 2}), {3, 4});
        const Tensor out = attention.forward(single);
        CHECK_NEAR(out.at({0, 0, 0}), 3.0, 1e-5);
        CHECK_NEAR(out.at({0, 0, 1}), 4.0, 1e-5);
    }

    // the real Qwen3-shaped layer, for shapes only
    {
        const size_t D = 64;
        const size_t H = 4;
        const size_t DH = 16;

        ScaledDotProductAttention attention(
            {Linear(Tensor{Shape({H * DH, D})}), Linear(Tensor{Shape({H * DH, D})}),
             Linear(Tensor{Shape({H * DH, D})}), Linear(Tensor{Shape({D, H * DH})})},
            {H, H, DH, std::nullopt});

        CHECK_EQ(attention.forward(Tensor{Shape({1, 8, D})}).shape(), Shape({1, 8, D}));
    }

    // constructor validation
    {
        const Tensor good = Tensor{Shape({4, 4})};

        // grouped-query attention: refused in E7, supported since E8.S4
        {
            const ScaledDotProductAttention grouped(
                {Linear(Tensor{Shape({4, 4})}), Linear(Tensor{Shape({2, 4})}),
                 Linear(Tensor{Shape({2, 4})}), Linear(Tensor{Shape({4, 4})})},
                {2, 1, 2, std::nullopt});
            CHECK_EQ(grouped.kv_heads(), size_t{1});
            CHECK_EQ(grouped.forward(Tensor{Shape({1, 3, 4})}).shape(), Shape({1, 3, 4}));
        }

        // but a head count that does not divide is still refused
        CHECK_THROWS_AS(ScaledDotProductAttention(
                            {Linear(Tensor{Shape({6, 4})}), Linear(Tensor{Shape({8, 4})}),
                             Linear(Tensor{Shape({8, 4})}), Linear(Tensor{Shape({4, 6})})},
                            {3, 4, 2, std::nullopt}),
                        std::invalid_argument);

        // q_proj must produce heads * head_dim
        CHECK_THROWS_AS(ScaledDotProductAttention({Linear(Tensor{Shape({3, 4})}), Linear(good),
                                                   Linear(good), Linear(good)},
                                                  {2, 2, 2, std::nullopt}),
                        std::invalid_argument);

        // o_proj must map back to the input width
        CHECK_THROWS_AS(ScaledDotProductAttention({Linear(good), Linear(good), Linear(good),
                                                   Linear(Tensor{Shape({3, 4})})},
                                                  {2, 2, 2, std::nullopt}),
                        std::invalid_argument);

        CHECK_THROWS_AS(ScaledDotProductAttention({Linear(good), Linear(good), Linear(good),
                                                   Linear(good)},
                                                  {0, 0, 2, std::nullopt}),
                        std::invalid_argument);
    }

    // AD2: no weight is copied
    {
        const Tensor q_weight = identity(2);
        ScaledDotProductAttention attention(
            {Linear(q_weight), Linear(identity(2)), Linear(identity(2)), Linear(identity(2))},
            {1, 1, 2, std::nullopt});
        CHECK_EQ(q_weight.storage().use_count(), long{2});
        CHECK_EQ(q_weight.at({0, 0}), 1.0f);
    }

    return VEDA_TEST_SUMMARY("AttentionLayerTest");
}
