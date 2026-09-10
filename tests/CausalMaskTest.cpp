// E7.S4.T5 — the causal mask

#include "Attention.h"
#include "Shape.h"
#include "Softmax.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::model::apply_causal_mask;
using veda::model::causal_mask;
using veda::ops::softmax;

namespace
{
Tensor ones(const Shape& shape)
{
    Tensor t{shape};
    for (size_t i = 0; i < shape.size(); ++i)
    {
        t.data()[i] = 1.0f;
    }
    return t;
}

const float infinity = std::numeric_limits<float>::infinity();
} // namespace

int main()
{
    // the lower triangle: 0 where allowed, -inf above
    {
        const Tensor mask = causal_mask(3, 3);
        CHECK_EQ(mask.shape(), Shape({3, 3}));

        CHECK_EQ(mask.at({0, 0}), 0.0f);
        CHECK_EQ(mask.at({0, 1}), -infinity);
        CHECK_EQ(mask.at({0, 2}), -infinity);
        CHECK_EQ(mask.at({1, 0}), 0.0f);
        CHECK_EQ(mask.at({1, 1}), 0.0f);
        CHECK_EQ(mask.at({1, 2}), -infinity);
        CHECK_EQ(mask.at({2, 0}), 0.0f);
        CHECK_EQ(mask.at({2, 2}), 0.0f);

        // the diagonal is always open — which is why the mask can never make a NaN row
        for (size_t t = 0; t < 3; ++t)
        {
            CHECK_EQ(mask.at({t, t}), 0.0f);
        }
    }

    // the cached-generation case: one new query sees the whole history
    {
        const Tensor mask = causal_mask(1, 4);
        CHECK_EQ(mask.shape(), Shape({1, 4}));
        for (size_t s = 0; s < 4; ++s)
        {
            CHECK_EQ(mask.at({0, s}), 0.0f);
        }

        // a partial prefill: 2 queries at the end of a 5-key history, so t=0 sits at position 3
        const Tensor partial = causal_mask(2, 5);
        CHECK_EQ(partial.at({0, 3}), 0.0f);
        CHECK_EQ(partial.at({0, 4}), -infinity);
        CHECK_EQ(partial.at({1, 4}), 0.0f);

        CHECK_THROWS_AS(causal_mask(4, 3), std::invalid_argument);
    }

    // *** the property of this story: after softmax the strict upper triangle is EXACTLY zero ***
    {
        const Tensor scores = ones(Shape({3, 3}));
        const Tensor masked = apply_causal_mask(scores);

        CHECK_EQ(masked.at({0, 1}), -infinity);
        CHECK_EQ(masked.at({1, 2}), -infinity);
        CHECK_EQ(masked.at({2, 0}), 1.0f);

        const Tensor weights = softmax(masked, 1);

        // exactly 0.0f, not 1e-38: anything else means the masking happened after softmax
        CHECK_EQ(weights.at({0, 1}), 0.0f);
        CHECK_EQ(weights.at({0, 2}), 0.0f);
        CHECK_EQ(weights.at({1, 2}), 0.0f);

        // row t: uniform over its first t+1 entries
        CHECK_EQ(weights.at({0, 0}), 1.0f);
        CHECK_NEAR(weights.at({1, 0}), 0.5, 1e-6);
        CHECK_NEAR(weights.at({1, 1}), 0.5, 1e-6);
        CHECK_NEAR(weights.at({2, 0}), 1.0 / 3.0, 1e-6);
        CHECK_NEAR(weights.at({2, 2}), 1.0 / 3.0, 1e-6);

        // every row sums to 1 — over its allowed prefix, and over the whole row
        for (size_t t = 0; t < 3; ++t)
        {
            float prefix = 0.0f;
            float whole = 0.0f;
            for (size_t s = 0; s < 3; ++s)
            {
                whole += weights.at({t, s});
                if (s <= t)
                {
                    prefix += weights.at({t, s});
                }
            }
            CHECK_NEAR(prefix, 1.0, 1e-6);
            CHECK_NEAR(whole, 1.0, 1e-6);
            CHECK_EQ(prefix, whole);   // nothing leaked past the diagonal
        }

        // no NaN anywhere: the diagonal is always open
        for (size_t t = 0; t < 3; ++t)
        {
            for (size_t s = 0; s < 3; ++s)
            {
                CHECK(!std::isnan(weights.at({t, s})));
            }
        }
    }

    // what zeroing after softmax would have cost: the row would no longer sum to 1
    {
        const Tensor unmasked_weights = softmax(ones(Shape({3, 3})), 1);
        float zeroed_row = 0.0f;
        for (size_t s = 0; s <= 1; ++s)   // row 1, keeping only its allowed prefix
        {
            zeroed_row += unmasked_weights.at({1, s});
        }
        CHECK_NEAR(zeroed_row, 2.0 / 3.0, 1e-6);   // not 1 — the output would be scaled down by 1/3
        CHECK(zeroed_row < 0.9f);
    }

    // and why addition rather than a 0/1 multiplication: a score of 0 is "neutral", not "forbidden"
    {
        Tensor multiplied{Shape({1, 3})};
        multiplied.data()[0] = 1.0f;   // allowed
        multiplied.data()[1] = 0.0f;   // "masked" by multiplying
        multiplied.data()[2] = 0.0f;
        const Tensor weights = softmax(multiplied, 1);
        CHECK(weights.at({0, 1}) > 0.2f);   // softmax gives the forbidden positions 21% each
        CHECK(weights.at({0, 1}) != 0.0f);
    }

    // broadcasting over batch and heads: one triangle, applied to every head
    {
        Tensor scores{Shape({1, 2, 3, 3})};
        for (size_t i = 0; i < scores.numel(); ++i)
        {
            scores.data()[i] = static_cast<float>(i % 3) + 1.0f;
        }
        const Tensor masked = apply_causal_mask(scores);
        CHECK_EQ(masked.shape(), Shape({1, 2, 3, 3}));

        const Tensor weights = softmax(masked, 3);
        for (size_t h = 0; h < 2; ++h)
        {
            CHECK_EQ(weights.at({0, h, 0, 1}), 0.0f);
            CHECK_EQ(weights.at({0, h, 0, 2}), 0.0f);
            CHECK_EQ(weights.at({0, h, 1, 2}), 0.0f);
            CHECK_EQ(weights.at({0, h, 0, 0}), 1.0f);

            float row = 0.0f;
            for (size_t s = 0; s < 3; ++s)
            {
                row += weights.at({0, h, 2, s});
            }
            CHECK_NEAR(row, 1.0, 1e-6);
        }

        // one output allocated, not B*H masks
        CHECK(masked.is_contiguous());
        CHECK(masked.storage() != scores.storage());
        CHECK_EQ(scores.at({0, 0, 0, 1}), 2.0f);   // the input is unmodified
    }

    // scores that already contain -inf stay -inf
    {
        Tensor scores = ones(Shape({2, 2}));
        scores.data()[0] = -infinity;
        const Tensor masked = apply_causal_mask(scores);
        CHECK_EQ(masked.at({0, 0}), -infinity);
        CHECK_EQ(masked.at({1, 0}), 1.0f);

        // a fully-masked row does produce NaN — which the causal mask alone never creates
        const Tensor weights = softmax(masked, 1);
        CHECK(std::isnan(weights.at({0, 0})));
    }

    // the real pipeline order: scores, scale, mask, softmax
    {
        const Tensor q = Tensor{Shape({1, 1, 3, 4})};
        const Tensor k = Tensor{Shape({1, 1, 3, 4})};
        const Tensor weights = softmax(
            apply_causal_mask(veda::model::scale_scores(veda::model::attention_scores(q, k), 4)), 3);

        CHECK_EQ(weights.shape(), Shape({1, 1, 3, 3}));
        CHECK_EQ(weights.at({0, 0, 0, 1}), 0.0f);
        CHECK_EQ(weights.at({0, 0, 0, 0}), 1.0f);
        CHECK_NEAR(weights.at({0, 0, 2, 1}), 1.0 / 3.0, 1e-6);
    }

    // edge cases
    {
        CHECK_EQ(causal_mask(1, 1).at({0, 0}), 0.0f);
        CHECK_EQ(causal_mask(0, 0).numel(), size_t{0});
        CHECK_EQ(causal_mask(0, 5).shape(), Shape({0, 5}));
        CHECK_EQ(apply_causal_mask(Tensor{Shape({0, 3})}).numel(), size_t{0});
        CHECK_THROWS_AS(apply_causal_mask(Tensor{Shape({3})}), std::invalid_argument);
    }

    return VEDA_TEST_SUMMARY("CausalMaskTest");
}
