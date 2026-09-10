// E8.S4.T4 — grouped-query attention

#include "Attention.h"
#include "Linear.h"
#include "ScaledDotProductAttention.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::model::expand_kv_heads;
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
    // *** THE GROUPING, NAMED: with H = 4 and Hkv = 2, query head 2 uses KV head 1 ***
    // Two plausible conventions produce identical shapes; only this assertion separates them.
    {
        const Tensor kv = filled(Shape({1, 2, 1, 2}), {10, 11, 20, 21});
        const Tensor expanded = expand_kv_heads(kv, 4);

        CHECK_EQ(expanded.shape(), Shape({1, 4, 1, 2}));

        CHECK_EQ(expanded.at({0, 0, 0, 0}), 10.0f);   // head 0 -> kv 0
        CHECK_EQ(expanded.at({0, 1, 0, 0}), 10.0f);   // head 1 -> kv 0
        CHECK_EQ(expanded.at({0, 2, 0, 0}), 20.0f);   // head 2 -> kv 1   <-- the assertion
        CHECK_EQ(expanded.at({0, 3, 0, 0}), 20.0f);   // head 3 -> kv 1
        CHECK_EQ(expanded.at({0, 2, 0, 1}), 21.0f);

        // the interleaved convention would have put kv 1 in head 1 — it does not
        CHECK(expanded.at({0, 1, 0, 0}) != kv.at({0, 1, 0, 0}));

        // stated as the formula, for every head
        const size_t repeats = 4 / 2;
        for (size_t h = 0; h < 4; ++h)
        {
            for (size_t d = 0; d < 2; ++d)
            {
                CHECK_EQ(expanded.at({0, h, 0, d}), kv.at({0, h / repeats, 0, d}));
            }
        }
    }

    // it materialises: head h reads kv head h/rep, which is not a constant stride
    {
        const Tensor kv = filled(Shape({1, 2, 1, 2}), {10, 11, 20, 21});
        const Tensor expanded = expand_kv_heads(kv, 4);
        CHECK(expanded.is_contiguous());
        CHECK(expanded.storage() != kv.storage());
        CHECK_EQ(kv.at({0, 0, 0, 0}), 10.0f);   // the input is unmodified
    }

    // the family: Hkv = H is plain multi-head, Hkv = 1 is multi-query
    {
        const Tensor kv = filled(Shape({1, 3, 1, 1}), {1, 2, 3});

        // H == Hkv: values unchanged, storage fresh
        const Tensor same = expand_kv_heads(kv, 3);
        CHECK_EQ(same.shape(), Shape({1, 3, 1, 1}));
        CHECK_EQ(same.at({0, 2, 0, 0}), 3.0f);
        CHECK(same.storage() != kv.storage());

        // rep = 2
        const Tensor doubled = expand_kv_heads(kv, 6);
        CHECK_EQ(doubled.shape(), Shape({1, 6, 1, 1}));
        CHECK_EQ(doubled.at({0, 0, 0, 0}), 1.0f);
        CHECK_EQ(doubled.at({0, 1, 0, 0}), 1.0f);
        CHECK_EQ(doubled.at({0, 2, 0, 0}), 2.0f);
        CHECK_EQ(doubled.at({0, 5, 0, 0}), 3.0f);

        // multi-query: one kv head serving all six
        const Tensor single = filled(Shape({1, 1, 1, 2}), {7, 8});
        const Tensor broadcast = expand_kv_heads(single, 6);
        CHECK_EQ(broadcast.shape(), Shape({1, 6, 1, 2}));
        for (size_t h = 0; h < 6; ++h)
        {
            CHECK_EQ(broadcast.at({0, h, 0, 0}), 7.0f);
        }
    }

    // a non-contiguous input, straight from split_heads
    {
        Tensor projected{Shape({1, 2, 4})};   // [B, T, Hkv*Dh] with Hkv = 2, Dh = 2
        for (size_t i = 0; i < projected.numel(); ++i)
        {
            projected.data()[i] = static_cast<float>(i);
        }
        const Tensor kv = veda::model::split_heads(projected, 2, 2);
        CHECK(!kv.is_contiguous());

        const Tensor expanded = expand_kv_heads(kv, 4);
        CHECK_EQ(expanded.shape(), Shape({1, 4, 2, 2}));
        for (size_t h = 0; h < 4; ++h)
        {
            for (size_t t = 0; t < 2; ++t)
            {
                for (size_t d = 0; d < 2; ++d)
                {
                    CHECK_EQ(expanded.at({0, h, t, d}), kv.at({0, h / 2, t, d}));
                }
            }
        }
    }

    // refusals and edge cases
    {
        const Tensor kv = filled(Shape({1, 2, 1, 2}), {10, 11, 20, 21});

        CHECK_THROWS_AS(expand_kv_heads(kv, 3), std::invalid_argument);   // 3 does not divide by 2
        CHECK_THROWS_AS(expand_kv_heads(kv, 0), std::invalid_argument);
        CHECK_THROWS_AS(expand_kv_heads(Tensor{Shape({2, 1, 2})}, 4), std::invalid_argument);
        try
        {
            (void)expand_kv_heads(kv, 5);
        }
        catch (const std::invalid_argument& error)
        {
            const std::string message = error.what();
            CHECK(message.find("5") != std::string::npos);
            CHECK(message.find("2 kv heads") != std::string::npos);
        }

        CHECK_EQ(expand_kv_heads(Tensor{Shape({1, 2, 0, 2})}, 4).shape(), Shape({1, 4, 0, 2}));
        CHECK_EQ(expand_kv_heads(Tensor{Shape({1, 2, 3, 1})}, 4).shape(), Shape({1, 4, 3, 1}));
    }

    // --- the layer at H = 4, Hkv = 2 ---------------------------------------------------------------
    {
        const size_t D = 8;
        const size_t H = 4;
        const size_t HKV = 2;
        const size_t DH = 2;

        ScaledDotProductAttention attention(
            {Linear(identity(D)),                          // q: [H*Dh, D] = [8, 8]
             Linear(Tensor{Shape({HKV * DH, D})}),         // k: [4, 8]
             Linear(Tensor{Shape({HKV * DH, D})}),         // v: [4, 8]
             Linear(identity(D))},                         // o: [D, H*Dh] = [8, 8]
            {H, HKV, DH, std::nullopt});

        CHECK_EQ(attention.heads(), size_t{4});
        CHECK_EQ(attention.kv_heads(), size_t{2});
        CHECK_EQ(attention.forward(Tensor{Shape({1, 3, D})}).shape(), Shape({1, 3, D}));
    }

    // two query heads in the same group, given identical queries, must attend identically —
    // because they share a key head
    {
        const size_t D = 8;   // H = 4 heads of Dh = 2
        const size_t H = 4;
        const size_t HKV = 2;
        const size_t DH = 2;

        // k and v give each kv head a distinguishable constant
        Tensor k_weight{Shape({HKV * DH, D})};
        for (size_t row = 0; row < HKV * DH; ++row)
        {
            k_weight.data()[row * D + row] = 1.0f;
        }

        ScaledDotProductAttention attention(
            {Linear(identity(D)), Linear(k_weight), Linear(k_weight), Linear(identity(D))},
            {H, HKV, DH, std::nullopt});

        // a token whose four query heads are pairwise identical: heads 0,1 see [1,1] and
        // heads 2,3 see [2,2]
        const Tensor x = filled(Shape({1, 2, D}), {1, 1, 1, 1, 2, 2, 2, 2,
                                                   1, 1, 1, 1, 2, 2, 2, 2});
        const Tensor out = attention.forward(x);
        CHECK_EQ(out.shape(), Shape({1, 2, D}));

        // heads 0 and 1 share kv head 0 and were given the same query, so their outputs agree
        CHECK_NEAR(out.at({0, 1, 0}), out.at({0, 1, 2}), 1e-5);
        CHECK_NEAR(out.at({0, 1, 1}), out.at({0, 1, 3}), 1e-5);

        // heads 2 and 3 share kv head 1, likewise
        CHECK_NEAR(out.at({0, 1, 4}), out.at({0, 1, 6}), 1e-5);
        CHECK_NEAR(out.at({0, 1, 5}), out.at({0, 1, 7}), 1e-5);
    }

    // what GQA buys, in bytes: the KV cache at Qwen3-0.6B's real numbers
    {
        const size_t Hkv = 8;
        const size_t H = 16;
        const size_t Dh = 128;
        const size_t T = 32768;
        const size_t layers = 28;

        const size_t with_gqa = 2 * Hkv * T * Dh * sizeof(float) * layers;
        const size_t without = 2 * H * T * Dh * sizeof(float) * layers;

        CHECK_EQ(with_gqa, size_t{7'516'192'768});     // 7.5 GB
        CHECK_EQ(without, with_gqa * 2);               // 15 GB without grouping
        CHECK_EQ(without / with_gqa, size_t{2});
    }

    return VEDA_TEST_SUMMARY("GqaTest");
}
