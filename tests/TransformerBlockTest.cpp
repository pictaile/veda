// E9.S2.T3 — TransformerBlock

#include "FeedForward.h"
#include "Linear.h"
#include "RMSNorm.h"
#include "ScaledDotProductAttention.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"
#include "TransformerBlock.h"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::model::FeedForward;
using veda::model::ScaledDotProductAttention;
using veda::model::TransformerBlock;
using veda::nn::Linear;
using veda::nn::RMSNorm;

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

Tensor ones(size_t n)
{
    Tensor t{Shape({n})};
    for (size_t i = 0; i < n; ++i)
    {
        t.data()[i] = 1.0f;
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

const float eps = 1e-6f;

// A block whose projections are all zero: both sub-layers output nothing.
TransformerBlock zeroed_block(size_t D, size_t F)
{
    return TransformerBlock({RMSNorm(ones(D), eps),
                             ScaledDotProductAttention(
                                 {Linear(Tensor{Shape({D, D})}), Linear(Tensor{Shape({D, D})}),
                                  Linear(Tensor{Shape({D, D})}), Linear(Tensor{Shape({D, D})})},
                                 {1, 1, D, std::nullopt}),
                             RMSNorm(ones(D), eps),
                             FeedForward({Linear(Tensor{Shape({F, D})}), Linear(Tensor{Shape({F, D})}),
                                          Linear(Tensor{Shape({D, F})})})});
}
} // namespace

int main()
{
    // *** THE IDENTITY TEST: with every projection zero, the block is the identity, exactly ***
    // Any bug that drops an `x +` fails this, and no other test in the epic would.
    {
        const size_t D = 4;
        const TransformerBlock block = zeroed_block(D, 6);

        const Tensor x = filled(Shape({1, 3, D}), {1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4});
        const Tensor out = block.forward(x);

        CHECK_EQ(out.shape(), x.shape());
        for (size_t t = 0; t < 3; ++t)
        {
            for (size_t d = 0; d < D; ++d)
            {
                CHECK_EQ(out.at({0, t, d}), x.at({0, t, d}));   // exactly, not nearly
            }
        }

        CHECK_EQ(block.hidden_size(), D);
    }

    // the same at other shapes, since the property is structural
    {
        const TransformerBlock block = zeroed_block(2, 3);
        for (const Shape& shape : {Shape({1, 1, 2}), Shape({2, 4, 2}), Shape({1, 8, 2})})
        {
            Tensor x{shape};
            for (size_t i = 0; i < x.numel(); ++i)
            {
                x.data()[i] = static_cast<float>(i) - 3.0f;
            }
            const Tensor out = block.forward(x);
            bool identical = true;
            for (size_t i = 0; i < x.numel(); ++i)
            {
                if (out.data()[i] != x.data()[i])
                {
                    identical = false;
                }
            }
            CHECK(identical);
        }
    }

    // the hand-computed case: only v_proj and o_proj non-zero, T = 1
    //   x = [1, 2]        RMS = sqrt((1+4)/2) = 1.5811
    //   norm(x) = [0.6325, 1.2649]
    //   T = 1, so the single attention weight is 1 and the output is v_proj(norm(x))
    //   h = x + that
    {
        const size_t D = 2;
        const TransformerBlock block(
            {RMSNorm(ones(D), eps),
             ScaledDotProductAttention({Linear(Tensor{Shape({D, D})}), Linear(Tensor{Shape({D, D})}),
                                        Linear(identity(D)), Linear(identity(D))},
                                       {1, 1, D, std::nullopt}),
             RMSNorm(ones(D), eps),
             FeedForward({Linear(Tensor{Shape({3, D})}), Linear(Tensor{Shape({3, D})}),
                          Linear(Tensor{Shape({D, 3})})})});

        const Tensor x = filled(Shape({1, 1, D}), {1, 2});
        const Tensor out = block.forward(x);

        // h = x + norm(x) = [1 + 0.6325, 2 + 1.2649]; the FFN is zero, so out = h
        CHECK_NEAR(out.at({0, 0, 0}), 1.6325, 1e-3);
        CHECK_NEAR(out.at({0, 0, 1}), 3.2649, 1e-3);
    }

    // *** the residual stream grows: |out| > |x| under pre-norm ***
    {
        const size_t D = 4;
        const TransformerBlock block(
            {RMSNorm(ones(D), eps),
             ScaledDotProductAttention({Linear(identity(D)), Linear(identity(D)), Linear(identity(D)),
                                        Linear(identity(D))},
                                       {1, 1, D, std::nullopt}),
             RMSNorm(ones(D), eps),
             FeedForward({Linear(identity(D)), Linear(identity(D)), Linear(identity(D))})});

        const Tensor x = filled(Shape({1, 2, D}), {1, 2, 3, 4, 4, 3, 2, 1});
        const Tensor out = block.forward(x);

        double before = 0.0;
        double after = 0.0;
        for (size_t i = 0; i < x.numel(); ++i)
        {
            before += static_cast<double>(x.data()[i]) * x.data()[i];
            after += static_cast<double>(out.data()[i]) * out.data()[i];
        }
        CHECK(std::sqrt(after) > std::sqrt(before));

        // which is fine: the next block normalises before reading, and the final norm cleans up
        CHECK(std::sqrt(after) < 10.0 * std::sqrt(before));
    }

    // the position reaches attention through the block
    {
        const size_t D = 4;
        const TransformerBlock rotated(
            {RMSNorm(ones(D), eps),
             ScaledDotProductAttention({Linear(identity(D)), Linear(identity(D)), Linear(identity(D)),
                                        Linear(identity(D))},
                                       {1, 1, D, 10000.0f}),
             RMSNorm(ones(D), eps),
             FeedForward({Linear(Tensor{Shape({3, D})}), Linear(Tensor{Shape({3, D})}),
                          Linear(Tensor{Shape({D, 3})})})});

        const TransformerBlock unrotated(
            {RMSNorm(ones(D), eps),
             ScaledDotProductAttention({Linear(identity(D)), Linear(identity(D)), Linear(identity(D)),
                                        Linear(identity(D))},
                                       {1, 1, D, std::nullopt}),
             RMSNorm(ones(D), eps),
             FeedForward({Linear(Tensor{Shape({3, D})}), Linear(Tensor{Shape({3, D})}),
                          Linear(Tensor{Shape({D, 3})})})});

        const Tensor x = filled(Shape({1, 3, D}), {1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0});

        bool differs = false;
        for (size_t d = 0; d < D; ++d)
        {
            if (std::fabs(rotated.forward(x).at({0, 2, d}) - unrotated.forward(x).at({0, 2, d})) >
                1e-5f)
            {
                differs = true;
            }
        }
        CHECK(differs);   // rotation changed the attention, and the block passed it through
    }

    // a block with GQA and QK-Norm — nothing here cares
    {
        const size_t D = 8;
        const size_t H = 4;
        const size_t HKV = 2;
        const size_t DH = 2;

        const TransformerBlock block(
            {RMSNorm(ones(D), eps),
             ScaledDotProductAttention({Linear(identity(D)), Linear(Tensor{Shape({HKV * DH, D})}),
                                        Linear(Tensor{Shape({HKV * DH, D})}), Linear(identity(D)),
                                        RMSNorm(ones(DH), eps), RMSNorm(ones(DH), eps)},
                                       {H, HKV, DH, 1000000.0f}),
             RMSNorm(ones(D), eps),
             FeedForward({Linear(Tensor{Shape({12, D})}), Linear(Tensor{Shape({12, D})}),
                          Linear(Tensor{Shape({D, 12})})})});

        CHECK_EQ(block.forward(Tensor{Shape({1, 5, D})}).shape(), Shape({1, 5, D}));
        CHECK_EQ(block.forward(Tensor{Shape({1, 5, D})}, 7).shape(), Shape({1, 5, D}));
    }

    // construction validation, and no hard-coded dimensions
    {
        // the two norms must agree
        CHECK_THROWS_AS(
            TransformerBlock({RMSNorm(ones(4), eps),
                              ScaledDotProductAttention({Linear(Tensor{Shape({4, 4})}),
                                                         Linear(Tensor{Shape({4, 4})}),
                                                         Linear(Tensor{Shape({4, 4})}),
                                                         Linear(Tensor{Shape({4, 4})})},
                                                        {1, 1, 4, std::nullopt}),
                              RMSNorm(ones(6), eps),
                              FeedForward({Linear(Tensor{Shape({6, 4})}), Linear(Tensor{Shape({6, 4})}),
                                           Linear(Tensor{Shape({4, 6})})})}),
            std::invalid_argument);

        // the feed-forward must take the same width
        CHECK_THROWS_AS(
            TransformerBlock({RMSNorm(ones(4), eps),
                              ScaledDotProductAttention({Linear(Tensor{Shape({4, 4})}),
                                                         Linear(Tensor{Shape({4, 4})}),
                                                         Linear(Tensor{Shape({4, 4})}),
                                                         Linear(Tensor{Shape({4, 4})})},
                                                        {1, 1, 4, std::nullopt}),
                              RMSNorm(ones(4), eps),
                              FeedForward({Linear(Tensor{Shape({6, 8})}), Linear(Tensor{Shape({6, 8})}),
                                           Linear(Tensor{Shape({8, 6})})})}),
            std::invalid_argument);
    }

    // the input is unmodified
    {
        const TransformerBlock block = zeroed_block(2, 3);
        const Tensor x = filled(Shape({1, 2, 2}), {1, 2, 3, 4});
        const Tensor out = block.forward(x);
        CHECK_EQ(x.at({0, 0, 0}), 1.0f);
        CHECK(out.storage() != x.storage());
    }

    return VEDA_TEST_SUMMARY("TransformerBlockTest");
}
