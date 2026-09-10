// E7.S5.T6 — core::contiguous, attention_context and merge_heads

#include "Attention.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <stdexcept>
#include <vector>

using veda::core::contiguous;
using veda::core::Shape;
using veda::core::Tensor;
using veda::model::attention_context;
using veda::model::merge_heads;
using veda::model::split_heads;

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
    // --- core::contiguous, whose first caller is the head merge ----------------------------------
    {
        const Tensor source = counting(Shape({2, 3}));      // 0..5
        const Tensor transposed = source.transpose(0, 1);   // (3,2), strides (1,3)
        CHECK(!transposed.is_contiguous());

        const Tensor dense = contiguous(transposed);
        CHECK(dense.is_contiguous());
        CHECK_EQ(dense.shape(), Shape({3, 2}));
        CHECK(dense.storage() != transposed.storage());

        // same values in index order — the buffer is now 0 3 1 4 2 5
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 2; ++j)
            {
                CHECK_EQ(dense.at({i, j}), transposed.at({i, j}));
            }
        }
        CHECK_EQ(dense.data()[1], 3.0f);
        CHECK_EQ(dense.data()[2], 1.0f);

        // an already-dense tensor is still copied: a caller wanting a dense tensor wants an
        // independent one
        const Tensor copy = contiguous(source);
        CHECK(copy.storage() != source.storage());
        CHECK_EQ(copy.at({1, 2}), source.at({1, 2}));

        // a sliced view, and an empty one
        CHECK_EQ(contiguous(source.slice(1, 1, 2)).at({0, 0}), 1.0f);
        CHECK_EQ(contiguous(Tensor{Shape({2, 0})}).numel(), size_t{0});
        CHECK_EQ(contiguous(filled(Shape({}), {7.0f})).at({}), 7.0f);
    }

    // --- attention_context: spending the weights --------------------------------------------------
    // weights from S4 (causal, uniform over the allowed prefix), values from S1
    {
        const Tensor weights = filled(Shape({1, 1, 3, 3}),
                                      {1.0f, 0.0f, 0.0f,
                                       0.5f, 0.5f, 0.0f,
                                       1.0f / 3, 1.0f / 3, 1.0f / 3});
        const Tensor values = filled(Shape({1, 1, 3, 2}), {10, 0, 0, 10, 5, 5});

        const Tensor context = attention_context(weights, values);
        CHECK_EQ(context.shape(), Shape({1, 1, 3, 2}));

        CHECK_NEAR(context.at({0, 0, 0, 0}), 10.0, 1e-5);   // position 0 sees only value 0
        CHECK_NEAR(context.at({0, 0, 0, 1}), 0.0, 1e-5);
        CHECK_NEAR(context.at({0, 0, 1, 0}), 5.0, 1e-5);    // half of value 0, half of value 1
        CHECK_NEAR(context.at({0, 0, 1, 1}), 5.0, 1e-5);
        CHECK_NEAR(context.at({0, 0, 2, 0}), 5.0, 1e-5);    // a third of each
        CHECK_NEAR(context.at({0, 0, 2, 1}), 5.0, 1e-5);

        CHECK(context.is_contiguous());
        CHECK_EQ(values.at({0, 0, 0, 0}), 10.0f);   // inputs unmodified

        // every output lies inside the convex hull of the value rows: attention interpolates and
        // can never extrapolate
        for (size_t t = 0; t < 3; ++t)
        {
            for (size_t d = 0; d < 2; ++d)
            {
                CHECK(context.at({0, 0, t, d}) >= 0.0f);
                CHECK(context.at({0, 0, t, d}) <= 10.0f);
            }
        }

        CHECK_THROWS_AS(attention_context(weights, Tensor{Shape({1, 1, 4, 2})}),
                        std::invalid_argument);
        CHECK_THROWS_AS(attention_context(Tensor{Shape({3, 3})}, values), std::invalid_argument);
    }

    // --- merge_heads ------------------------------------------------------------------------------
    {
        // [1,2,2,2]: head 0 [[a0 a1],[c0 c1]], head 1 [[b0 b1],[d0 d1]]
        const Tensor per_head = filled(Shape({1, 2, 2, 2}), {0, 1, 4, 5, 2, 3, 6, 7});
        const Tensor merged = merge_heads(per_head);

        CHECK_EQ(merged.shape(), Shape({1, 2, 4}));
        CHECK_EQ(merged.at({0, 0, 0}), 0.0f);   // token 0: a0 a1 b0 b1
        CHECK_EQ(merged.at({0, 0, 1}), 1.0f);
        CHECK_EQ(merged.at({0, 0, 2}), 2.0f);
        CHECK_EQ(merged.at({0, 0, 3}), 3.0f);
        CHECK_EQ(merged.at({0, 1, 0}), 4.0f);   // token 1: c0 c1 d0 d1
        CHECK_EQ(merged.at({0, 1, 3}), 7.0f);

        // this is the one materialising copy in the attention path
        CHECK(merged.is_contiguous());
        CHECK(merged.storage() != per_head.storage());

        CHECK_THROWS_AS(merge_heads(Tensor{Shape({2, 3, 4})}), std::invalid_argument);
    }

    // *** the round trip: merge_heads(split_heads(x)) == x ***
    // A wrong transpose axis passes every shape assertion and fails this.
    {
        for (const std::vector<size_t>& layout : std::vector<std::vector<size_t>>{
                 {1, 3, 2, 2}, {2, 5, 4, 3}, {1, 8, 1, 6}, {1, 2, 6, 1}})
        {
            const size_t B = layout[0];
            const size_t T = layout[1];
            const size_t H = layout[2];
            const size_t DH = layout[3];

            const Tensor x = counting(Shape({B, T, H * DH}));
            const Tensor round_trip = merge_heads(split_heads(x, H, DH));

            CHECK_EQ(round_trip.shape(), x.shape());
            bool identical = true;
            for (size_t b = 0; b < B && identical; ++b)
            {
                for (size_t t = 0; t < T && identical; ++t)
                {
                    for (size_t c = 0; c < H * DH; ++c)
                    {
                        if (round_trip.at({b, t, c}) != x.at({b, t, c}))
                        {
                            identical = false;
                            break;
                        }
                    }
                }
            }
            CHECK(identical);
        }
    }

    // the shapes the model uses, and the size of the copy
    {
        const Tensor context{Shape({1, 16, 8, 128})};
        CHECK_EQ(merge_heads(context).shape(), Shape({1, 8, 2048}));

        // 4 MB per layer at T = 1024, H = 16, Dh = 128
        CHECK_EQ(size_t{1} * 1024 * 16 * 128 * sizeof(float), size_t{8'388'608});

        CHECK_EQ(merge_heads(Tensor{Shape({1, 1, 0, 4})}).shape(), Shape({1, 0, 4}));
        CHECK_EQ(merge_heads(Tensor{Shape({1, 4, 3, 1})}).shape(), Shape({1, 3, 4}));
    }

    return VEDA_TEST_SUMMARY("AttentionOutputTest");
}
