// E16.S2.T2 — reshape, transpose and RoPE on the tape

#include "GradientCheck.h"
#include "Rope.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"
#include "Variable.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

using veda::autograd::check_gradient;
using veda::autograd::mul;
using veda::autograd::reshape;
using veda::autograd::rope;
using veda::autograd::sum;
using veda::autograd::transpose;
using veda::autograd::Variable;
using veda::core::Shape;
using veda::core::Tensor;

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
    std::uniform_real_distribution<float> values(-1.5f, 1.5f);
    Tensor t{shape};
    for (size_t i = 0; i < shape.size(); ++i)
    {
        t.data()[i] = values(engine);
    }
    return t;
}
} // namespace

int main()
{
    std::mt19937 engine(20260913);

    // --- reshape ------------------------------------------------------------------------------------
    {
        // Non-square throughout: a [3,3] test passes with the inverse applied the wrong way round.
        const Tensor point = random_tensor(Shape({2, 6}), engine);
        const Tensor scale = random_tensor(Shape({2, 3, 2}), engine);
        const Variable weights = Variable::leaf(scale, false);

        Variable x = Variable::leaf(point);
        Variable reshaped = reshape(x, Shape({2, 3, 2}));
        CHECK_EQ(reshaped.value().shape(), Shape({2, 3, 2}));
        CHECK_EQ(reshaped.value().at({0, 1, 1}), point.at({0, 3}));

        sum(mul(reshaped, weights)).backward();
        CHECK_EQ(x.grad().shape(), Shape({2, 6}));
        CHECK_NEAR(x.grad().at({0, 3}), scale.at({0, 1, 1}), 1e-6);

        CHECK(check_gradient(
                  [&](const Variable& v) {
                      return sum(mul(reshape(v, Shape({2, 3, 2})), weights));
                  },
                  point)
                  .ok);
    }

    // --- transpose ------------------------------------------------------------------------------------
    {
        const Tensor point = filled(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
        Variable x = Variable::leaf(point);
        Variable swapped = transpose(x, 0, 1);

        CHECK_EQ(swapped.value().shape(), Shape({3, 2}));
        CHECK_EQ(swapped.value().at({2, 0}), 3.0f);
        CHECK(swapped.value().is_contiguous());   // materialised, not a view

        const Tensor scale = filled(Shape({3, 2}), {10, 20, 30, 40, 50, 60});
        sum(mul(swapped, Variable::leaf(scale, false))).backward();

        CHECK_EQ(x.grad().shape(), Shape({2, 3}));
        CHECK_EQ(x.grad().at({0, 0}), 10.0f);
        // dL/dx[i,j] = scale[j,i] — and this is the entry that tells the two directions apart:
        // reading scale[0,2] instead would give 30, which is a real number in the right place.
        CHECK_EQ(x.grad().at({0, 2}), 50.0f);
        CHECK_EQ(x.grad().at({1, 0}), 20.0f);

        // twice on the same axes is the identity, forward and backward
        Variable y = Variable::leaf(point);
        Variable twice = transpose(transpose(y, 0, 1), 0, 1);
        CHECK_EQ(twice.value().shape(), Shape({2, 3}));
        sum(mul(twice, Variable::leaf(point, false))).backward();
        CHECK_NEAR(y.grad().at({1, 2}), point.at({1, 2}), 1e-6);

        // the axes attention actually swaps
        const Tensor deep = random_tensor(Shape({2, 5, 3, 4}), engine);
        const Tensor deep_scale = random_tensor(Shape({2, 3, 5, 4}), engine);
        const Variable deep_weights = Variable::leaf(deep_scale, false);
        CHECK(check_gradient(
                  [&](const Variable& v) { return sum(mul(transpose(v, 1, 2), deep_weights)); },
                  deep)
                  .ok);
    }

    // --- rope ------------------------------------------------------------------------------------------
    {
        // A quarter turn, built by hand: cos = 0, sin = 1 at the only position.
        Tensor cosines{Shape({1, 2})};
        Tensor sines{Shape({1, 2})};
        cosines.data()[0] = 0.0f;
        cosines.data()[1] = 0.0f;
        sines.data()[0] = 1.0f;
        sines.data()[1] = 1.0f;

        Variable x = Variable::leaf(filled(Shape({1, 2}), {1.0f, 0.0f}));
        Variable rotated = rope(x, cosines, sines);

        // out[0] = 1*0 - 0*1 = 0;  out[1] = 0*0 + 1*1 = 1
        CHECK_NEAR(rotated.value().at({0, 0}), 0.0, 1e-6);
        CHECK_NEAR(rotated.value().at({0, 1}), 1.0, 1e-6);

        sum(mul(rotated, Variable::leaf(filled(Shape({1, 2}), {1.0f, 0.0f}), false))).backward();
        // dL/dx[0] = G[0]cos + G[1]sin = 0;  dL/dx[1] = G[1]cos - G[0]sin = -1
        CHECK_NEAR(x.grad().at({0, 0}), 0.0, 1e-6);
        CHECK_NEAR(x.grad().at({0, 1}), -1.0, 1e-6);
    }

    // *** a rotation preserves length, so the gradient's norm survives it ***
    {
        // One property catching a sign error, a swapped pair and a wrong table lookup at once.
        const veda::model::RopeTables tables = veda::model::rope_tables(6, 8, 10000.0f);
        const Tensor point = random_tensor(Shape({2, 6, 8}), engine);
        const Tensor incoming = random_tensor(Shape({2, 6, 8}), engine);

        Variable x = Variable::leaf(point);
        sum(mul(rope(x, tables.cosines, tables.sines), Variable::leaf(incoming, false)))
            .backward();

        double before = 0.0;
        double after = 0.0;
        for (size_t i = 0; i < incoming.numel(); ++i)
        {
            before += static_cast<double>(incoming.data()[i]) * incoming.data()[i];
            after += static_cast<double>(x.grad().data()[i]) * x.grad().data()[i];
        }
        CHECK_NEAR(std::sqrt(after), std::sqrt(before), 1e-3);

        // and the forward preserves it too — that is what a rotation is
        double rotated_norm = 0.0;
        double original_norm = 0.0;
        const Tensor rotated = veda::model::apply_rope(point, tables);
        for (size_t i = 0; i < point.numel(); ++i)
        {
            original_norm += static_cast<double>(point.data()[i]) * point.data()[i];
            rotated_norm += static_cast<double>(rotated.data()[i]) * rotated.data()[i];
        }
        CHECK_NEAR(std::sqrt(rotated_norm), std::sqrt(original_norm), 1e-3);

        // finite differences
        const Variable weights = Variable::leaf(incoming, false);
        CHECK(check_gradient(
                  [&](const Variable& v) {
                      return sum(mul(rope(v, tables.cosines, tables.sines), weights));
                  },
                  point)
                  .ok);
    }

    // *** the head split of E8, composed and differentiated end to end ***
    {
        const size_t B = 2, T = 3, H = 2, Dh = 4;
        const veda::model::RopeTables tables = veda::model::rope_tables(T, Dh, 10000.0f);

        const Tensor point = random_tensor(Shape({B, T, H * Dh}), engine);
        const Tensor scale = random_tensor(Shape({B, H, T, Dh}), engine);
        const Variable weights = Variable::leaf(scale, false);

        auto head_split = [&](const Variable& v) {
            Variable heads = transpose(reshape(v, Shape({B, T, H, Dh})), 1, 2);
            return sum(mul(rope(heads, tables.cosines, tables.sines), weights));
        };

        Variable x = Variable::leaf(point);
        Variable loss = head_split(x);
        CHECK_EQ(x.grad().shape(), Shape({B, T, H * Dh}));
        loss.backward();
        CHECK_EQ(x.grad().shape(), Shape({B, T, H * Dh}));

        CHECK(check_gradient(head_split, point).ok);
    }

    // --- refusals and the no-tape path -----------------------------------------------------------------
    {
        const Variable frozen = Variable::leaf(random_tensor(Shape({2, 4}), engine), false);
        CHECK(!reshape(frozen, Shape({4, 2})).requires_grad());
        CHECK(!transpose(frozen, 0, 1).requires_grad());

        const veda::model::RopeTables tables = veda::model::rope_tables(2, 4, 10000.0f);
        CHECK(!rope(frozen, tables.cosines, tables.sines).requires_grad());
        CHECK(transpose(frozen, 0, 1).node()->parents.empty());

        // mismatched tables, and a rank too low
        const Variable x = Variable::leaf(random_tensor(Shape({2, 4}), engine), false);
        const veda::model::RopeTables wrong = veda::model::rope_tables(5, 4, 10000.0f);
        CHECK_THROWS_AS(rope(x, wrong.cosines, wrong.sines), std::invalid_argument);

        const Variable flat = Variable::leaf(random_tensor(Shape({4}), engine), false);
        CHECK_THROWS_AS(rope(flat, tables.cosines, tables.sines), std::invalid_argument);
    }

    return VEDA_TEST_SUMMARY("AutogradViewsTest");
}
