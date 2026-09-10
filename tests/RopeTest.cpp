// E8.S2.T2 — apply_rope

#include "Attention.h"
#include "Rope.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::model::apply_rope;
using veda::model::rope_tables;
using veda::model::RopeTables;

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
    std::normal_distribution<float> values(0.0f, 1.0f);
    Tensor t{shape};
    for (size_t i = 0; i < shape.size(); ++i)
    {
        t.data()[i] = values(engine);
    }
    return t;
}

double norm_of_row(const Tensor& t, size_t position, size_t head_dim)
{
    double sum = 0.0;
    for (size_t d = 0; d < head_dim; ++d)
    {
        const double value = t.at({0, 0, position, d});
        sum += value * value;
    }
    return std::sqrt(sum);
}

double dot_of_rows(const Tensor& a, size_t pa, const Tensor& b, size_t pb, size_t head_dim)
{
    double sum = 0.0;
    for (size_t d = 0; d < head_dim; ++d)
    {
        sum += static_cast<double>(a.at({0, 0, pa, d})) * b.at({0, 0, pb, d});
    }
    return sum;
}
} // namespace

int main()
{
    // the tables: position 0 rotates by nothing
    {
        const RopeTables tables = rope_tables(1, 4, 10000.0f);
        CHECK_EQ(tables.cosines.shape(), Shape({1, 4}));
        for (size_t d = 0; d < 4; ++d)
        {
            CHECK_EQ(tables.cosines.at({0, d}), 1.0f);
            CHECK_EQ(tables.sines.at({0, d}), 0.0f);
        }
    }

    // the frequency ladder, and each angle repeated across both halves — the pairing convention
    {
        const RopeTables tables = rope_tables(3, 4, 10000.0f);

        // omega = (1, 0.01), so at position 1 the angles are (1.0, 0.01)
        CHECK_NEAR(tables.cosines.at({1, 0}), std::cos(1.0), 1e-6);
        CHECK_NEAR(tables.sines.at({1, 0}), std::sin(1.0), 1e-6);
        CHECK_NEAR(tables.cosines.at({1, 1}), std::cos(0.01), 1e-6);
        CHECK_NEAR(tables.sines.at({1, 1}), std::sin(0.01), 1e-6);

        // channel i and channel i + Dh/2 share an angle: that IS the half-split pairing
        CHECK_EQ(tables.cosines.at({1, 0}), tables.cosines.at({1, 2}));
        CHECK_EQ(tables.cosines.at({1, 1}), tables.cosines.at({1, 3}));
        CHECK_EQ(tables.sines.at({2, 0}), tables.sines.at({2, 2}));

        // position 2 is twice the angle of position 1
        CHECK_NEAR(tables.cosines.at({2, 0}), std::cos(2.0), 1e-6);
    }

    // the hand-computed case, which pins the convention down
    {
        const RopeTables tables = rope_tables(2, 4, 10000.0f);
        const Tensor x = filled(Shape({1, 1, 2, 4}), {1, 0, 0, 0,     // position 0
                                                      1, 0, 0, 0});   // position 1
        const Tensor out = apply_rope(x, tables);

        CHECK_EQ(out.shape(), Shape({1, 1, 2, 4}));

        // position 0: unchanged, exactly
        CHECK_EQ(out.at({0, 0, 0, 0}), 1.0f);
        CHECK_EQ(out.at({0, 0, 0, 1}), 0.0f);
        CHECK_EQ(out.at({0, 0, 0, 2}), 0.0f);

        // position 1: channel 0 pairs with channel 2, rotated by 1 radian
        CHECK_NEAR(out.at({0, 0, 1, 0}), 0.5403, 1e-4);   // cos(1)
        CHECK_NEAR(out.at({0, 0, 1, 1}), 0.0, 1e-6);
        CHECK_NEAR(out.at({0, 0, 1, 2}), 0.8415, 1e-4);   // sin(1)
        CHECK_NEAR(out.at({0, 0, 1, 3}), 0.0, 1e-6);

        CHECK(out.is_contiguous());
        CHECK_EQ(x.at({0, 0, 1, 0}), 1.0f);   // the input is unmodified
    }

    // *** property 1: rotation preserves the norm ***
    {
        std::mt19937 engine(20260908);
        for (const size_t head_dim : {2u, 8u, 64u})
        {
            const size_t T = 5;
            const Tensor x = random_tensor(Shape({1, 1, T, head_dim}), engine);
            const Tensor out = apply_rope(x, rope_tables(T, head_dim, 1000000.0f));

            for (size_t p = 0; p < T; ++p)
            {
                CHECK_NEAR(norm_of_row(out, p, head_dim), norm_of_row(x, p, head_dim), 1e-4);
            }
        }
    }

    // *** property 2: position 0 is unchanged, exactly ***
    {
        std::mt19937 engine(271828);
        const size_t head_dim = 8;
        const Tensor x = random_tensor(Shape({1, 1, 1, head_dim}), engine);
        const Tensor out = apply_rope(x, rope_tables(1, head_dim, 10000.0f));

        for (size_t d = 0; d < head_dim; ++d)
        {
            CHECK_EQ(out.at({0, 0, 0, d}), x.at({0, 0, 0, d}));
        }
    }

    // *** property 3: the score depends only on the distance — the point of the whole task ***
    {
        std::mt19937 engine(31415);
        const size_t head_dim = 16;
        const float theta = 10000.0f;

        const Tensor q = random_tensor(Shape({1, 1, 1, head_dim}), engine);
        const Tensor k = random_tensor(Shape({1, 1, 1, head_dim}), engine);

        auto score_at = [&](size_t q_position, size_t k_position) {
            const Tensor rotated_q = apply_rope(q, rope_tables(1, head_dim, theta, q_position));
            const Tensor rotated_k = apply_rope(k, rope_tables(1, head_dim, theta, k_position));
            return dot_of_rows(rotated_q, 0, rotated_k, 0, head_dim);
        };

        // same distance, wildly different absolute positions
        CHECK_NEAR(score_at(5, 3), score_at(12, 10), 1e-3);
        CHECK_NEAR(score_at(5, 3), score_at(105, 103), 1e-3);
        CHECK_NEAR(score_at(1, 0), score_at(51, 50), 1e-3);

        // distance 0 is the unrotated dot product
        CHECK_NEAR(score_at(7, 7), dot_of_rows(q, 0, k, 0, head_dim), 1e-3);

        // a different distance gives a different score
        CHECK(std::fabs(score_at(5, 3) - score_at(5, 0)) > 1e-2);
    }

    // the position offset: rope_tables(1, Dh, theta, 5) is row 5 of rope_tables(6, Dh, theta)
    {
        const RopeTables whole = rope_tables(6, 8, 10000.0f);
        const RopeTables shifted = rope_tables(1, 8, 10000.0f, 5);
        for (size_t d = 0; d < 8; ++d)
        {
            CHECK_EQ(shifted.cosines.at({0, d}), whole.cosines.at({5, d}));
            CHECK_EQ(shifted.sines.at({0, d}), whole.sines.at({5, d}));
        }
    }

    // theta: a larger base slows the slow pairs down
    {
        const RopeTables small = rope_tables(2, 8, 10000.0f);
        const RopeTables large = rope_tables(2, 8, 1000000.0f);

        // the fast pair (channel 0) is identical — omega_0 is 1 regardless of theta
        CHECK_EQ(small.cosines.at({1, 0}), large.cosines.at({1, 0}));

        // the slow pair rotates less with the larger base
        CHECK(std::fabs(large.sines.at({1, 3})) < std::fabs(small.sines.at({1, 3})));
    }

    // a non-contiguous input, straight from split_heads
    {
        const size_t head_dim = 4;
        Tensor projected{Shape({1, 2, 2 * head_dim})};   // [B, T, H*Dh] with H = 2
        for (size_t i = 0; i < projected.numel(); ++i)
        {
            projected.data()[i] = static_cast<float>(i + 1);
        }
        const Tensor heads = veda::model::split_heads(projected, 2, head_dim);
        CHECK(!heads.is_contiguous());

        const Tensor out = apply_rope(heads, rope_tables(2, head_dim, 10000.0f));
        CHECK_EQ(out.shape(), Shape({1, 2, 2, head_dim}));
        CHECK(out.is_contiguous());

        // position 0 of both heads is untouched, and each head kept its own channels
        for (size_t h = 0; h < 2; ++h)
        {
            for (size_t d = 0; d < head_dim; ++d)
            {
                CHECK_EQ(out.at({0, h, 0, d}), heads.at({0, h, 0, d}));
            }
        }

        // and the two heads were rotated independently but by the same angles
        CHECK_NEAR(out.at({0, 0, 1, 0}),
                   heads.at({0, 0, 1, 0}) * std::cos(1.0) - heads.at({0, 0, 1, 2}) * std::sin(1.0),
                   1e-4);
    }

    // refusals and edge cases
    {
        CHECK_THROWS_AS(rope_tables(2, 3, 10000.0f), std::invalid_argument);   // odd head_dim
        CHECK_THROWS_AS(rope_tables(2, 0, 10000.0f), std::invalid_argument);
        CHECK_THROWS_AS(rope_tables(2, 4, 0.0f), std::invalid_argument);

        const RopeTables tables = rope_tables(3, 4, 10000.0f);
        CHECK_THROWS_AS(apply_rope(Tensor{Shape({1, 1, 2, 4})}, tables), std::invalid_argument);
        CHECK_THROWS_AS(apply_rope(Tensor{Shape({1, 1, 3, 8})}, tables), std::invalid_argument);
        CHECK_THROWS_AS(apply_rope(Tensor{Shape({4})}, tables), std::invalid_argument);

        // Dh = 2: a single pair
        const Tensor single = filled(Shape({1, 1, 1, 2}), {1, 0});
        CHECK_EQ(apply_rope(single, rope_tables(1, 2, 10000.0f)).at({0, 0, 0, 0}), 1.0f);

        // an empty sequence
        CHECK_EQ(apply_rope(Tensor{Shape({1, 1, 0, 4})}, rope_tables(0, 4, 10000.0f)).numel(),
                 size_t{0});
    }

    return VEDA_TEST_SUMMARY("RopeTest");
}
