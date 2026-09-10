// E14.S3.T4 — the non-linear gradients

#include "GradientCheck.h"
#include "Shape.h"
#include "Elementwise.h"
#include "Softmax.h"
#include "TestSupport.h"
#include "Tensor.h"
#include "Variable.h"

#include <cmath>
#include <random>
#include <vector>

using veda::autograd::check_gradient;
using veda::autograd::embedding;
using veda::autograd::gelu;
using veda::autograd::mul;
using veda::autograd::rms_normalize;
using veda::autograd::silu;
using veda::autograd::softmax;
using veda::autograd::sum;
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
    std::mt19937 engine(20260910);

    // --- silu ------------------------------------------------------------------------------------
    {
        Variable x = Variable::leaf(filled(Shape({3}), {0.0f, 1.0f, -1.0f}));
        sum(silu(x)).backward();

        // d/dx = s(1 + x(1-s));  at 0: 0.5;  at 1: 0.7311(1+0.2689) = 0.9277
        CHECK_NEAR(x.grad().at({0}), 0.5, 1e-4);
        CHECK_NEAR(x.grad().at({1}), 0.9277, 1e-3);
        CHECK_NEAR(x.grad().at({2}), 0.0723, 1e-3);   // symmetric partner of the above
        CHECK_EQ(x.grad().shape(), Shape({3}));

        const Tensor point = random_tensor(Shape({6}), engine);
        CHECK(check_gradient([](const Variable& v) { return sum(silu(v)); }, point).ok);

        const Tensor scale = random_tensor(Shape({6}), engine);
        const Variable weights = Variable::leaf(scale, false);
        CHECK(check_gradient([&](const Variable& v) { return sum(mul(silu(v), weights)); }, point)
                  .ok);
    }

    // --- gelu ------------------------------------------------------------------------------------
    {
        Variable x = Variable::leaf(filled(Shape({3}), {0.0f, 1.0f, -1.0f}));
        sum(gelu(x)).backward();

        CHECK_NEAR(x.grad().at({0}), 0.5, 1e-4);
        CHECK_NEAR(x.grad().at({1}), 1.0830, 1e-3);
        CHECK_NEAR(x.grad().at({2}), -0.0830, 1e-3);

        const Tensor point = random_tensor(Shape({6}), engine);
        CHECK(check_gradient([](const Variable& v) { return sum(gelu(v)); }, point).ok);
    }

    // --- softmax: the linear form, and what it implies ---------------------------------------------
    {
        // y = softmax([1,2,3]) = [0.0900, 0.2447, 0.6652];  G = [1,0,0]
        // sum G_j y_j = 0.0900;  dL/dx = y * (G - 0.0900)
        Variable x = Variable::leaf(filled(Shape({3}), {1.0f, 2.0f, 3.0f}));
        Variable y = softmax(x, 0);

        const Tensor picker = filled(Shape({3}), {1.0f, 0.0f, 0.0f});
        sum(mul(y, Variable::leaf(picker, false))).backward();

        CHECK_NEAR(x.grad().at({0}), 0.0819, 1e-3);
        CHECK_NEAR(x.grad().at({1}), -0.0220, 1e-3);
        CHECK_NEAR(x.grad().at({2}), -0.0599, 1e-3);

        // *** the gradient sums to zero along the lane ***
        // softmax's outputs are constrained to sum to 1, so moving one up must move the others down
        float total = 0.0f;
        for (size_t i = 0; i < 3; ++i)
        {
            total += x.grad().at({i});
        }
        CHECK_NEAR(total, 0.0, 1e-5);
    }

    // finite differences, and along different axes
    {
        const Tensor point = random_tensor(Shape({5}), engine);
        const Tensor scale = random_tensor(Shape({5}), engine);
        const Variable weights = Variable::leaf(scale, false);
        CHECK(check_gradient(
                  [&](const Variable& v) { return sum(mul(softmax(v, 0), weights)); }, point)
                  .ok);

        const Tensor matrix = random_tensor(Shape({3, 4}), engine);
        const Tensor matrix_scale = random_tensor(Shape({3, 4}), engine);
        const Variable matrix_weights = Variable::leaf(matrix_scale, false);

        CHECK(check_gradient(
                  [&](const Variable& v) { return sum(mul(softmax(v, 1), matrix_weights)); },
                  matrix)
                  .ok);
        CHECK(check_gradient(
                  [&](const Variable& v) { return sum(mul(softmax(v, 0), matrix_weights)); },
                  matrix)
                  .ok);

        // every lane's gradient sums to zero, whichever axis
        Variable v = Variable::leaf(matrix);
        sum(mul(softmax(v, 1), matrix_weights)).backward();
        for (size_t row = 0; row < 3; ++row)
        {
            float total = 0.0f;
            for (size_t column = 0; column < 4; ++column)
            {
                total += v.grad().at({row, column});
            }
            CHECK_NEAR(total, 0.0, 1e-4);
        }
    }

    // --- rms_normalize: the coupling term ----------------------------------------------------------
    {
        // A non-constant lane: a constant one makes the coupling term vanish, and the test would
        // pass without it.
        const Tensor point = filled(Shape({4}), {1.0f, -2.0f, 0.5f, 3.0f});
        const Tensor scale = random_tensor(Shape({4}), engine);
        const Variable weights = Variable::leaf(scale, false);

        CHECK(check_gradient(
                  [&](const Variable& v) { return sum(mul(rms_normalize(v, 1e-6f), weights)); },
                  point, 1e-2f, 3e-2)
                  .ok);

        // and on the shape the model uses, [B, T, D], lane by lane
        const Tensor batched = random_tensor(Shape({2, 3, 4}), engine);
        const Tensor batched_scale = random_tensor(Shape({2, 3, 4}), engine);
        const Variable batched_weights = Variable::leaf(batched_scale, false);
        CHECK(check_gradient(
                  [&](const Variable& v) {
                      return sum(mul(rms_normalize(v, 1e-6f), batched_weights));
                  },
                  batched, 1e-2f, 3e-2)
                  .ok);

        // *** the coupling matters: without it the gradient of sum(rms_normalize(x)) would be
        //     G/r alone, which is a different number ***
        Variable v = Variable::leaf(point);
        sum(rms_normalize(v, 0.0f)).backward();

        float squares = 0.0f;
        for (size_t i = 0; i < 4; ++i)
        {
            squares += point.at({i}) * point.at({i});
        }
        const float r = std::sqrt(squares / 4.0f);
        CHECK(std::fabs(v.grad().at({0}) - 1.0f / r) > 1e-3f);   // not simply G/r
    }

    // --- embedding: a scatter-add ------------------------------------------------------------------
    {
        const Tensor table_value = random_tensor(Shape({4, 3}), engine);
        const std::vector<int64_t> ids = {2, 0, 2};

        Variable table = Variable::leaf(table_value);
        Variable rows = embedding(table, ids, Shape({1, 3}));
        CHECK_EQ(rows.value().shape(), Shape({1, 3, 3}));
        CHECK_EQ(rows.value().at({0, 0, 1}), table_value.at({2, 1}));
        CHECK_EQ(rows.value().at({0, 1, 2}), table_value.at({0, 2}));

        const Tensor scale = filled(Shape({1, 3, 3}), {1, 1, 1, 2, 2, 2, 4, 4, 4});
        sum(mul(rows, Variable::leaf(scale, false))).backward();

        CHECK_EQ(table.grad().shape(), Shape({4, 3}));

        // *** repeated ids accumulate: row 2 was used twice, with weights 1 and 4 ***
        CHECK_NEAR(table.grad().at({2, 0}), 5.0, 1e-5);
        CHECK_NEAR(table.grad().at({2, 2}), 5.0, 1e-5);
        CHECK_NEAR(table.grad().at({0, 0}), 2.0, 1e-5);

        // rows nothing referenced get nothing
        CHECK_EQ(table.grad().at({1, 0}), 0.0f);
        CHECK_EQ(table.grad().at({3, 2}), 0.0f);

        // finite differences on the table
        const Tensor bigger_scale = random_tensor(Shape({1, 3, 3}), engine);
        const Variable weights = Variable::leaf(bigger_scale, false);
        CHECK(check_gradient(
                  [&](const Variable& v) {
                      return sum(mul(embedding(v, ids, Shape({1, 3})), weights));
                  },
                  table_value)
                  .ok);
    }

    // --- no requires_grad means no tape, for all five ------------------------------------------------
    {
        const Variable frozen = Variable::leaf(random_tensor(Shape({4}), engine), false);
        CHECK(!silu(frozen).requires_grad());
        CHECK(!gelu(frozen).requires_grad());
        CHECK(!softmax(frozen, 0).requires_grad());
        CHECK(!rms_normalize(frozen, 1e-6f).requires_grad());

        const Variable frozen_table = Variable::leaf(random_tensor(Shape({4, 2}), engine), false);
        CHECK(!embedding(frozen_table, {1, 2}, Shape({1, 2})).requires_grad());
        CHECK(silu(frozen).node()->parents.empty());
    }

    // --- a two-layer network learns a toy problem: the epic's milestone ---------------------------
    {
        // Fit y = silu(x . W1) . W2 to a constant target by gradient descent. If every gradient in
        // this epic is right, the loss falls; if any is wrong by a factor, it falls differently or
        // not at all.
        const size_t in = 3;
        const size_t hidden = 4;
        const size_t out = 2;

        Tensor w1_value = random_tensor(Shape({hidden, in}), engine);
        Tensor w2_value = random_tensor(Shape({out, hidden}), engine);
        const Tensor x_value = filled(Shape({1, in}), {0.5f, -1.0f, 2.0f});
        const Tensor target = filled(Shape({1, out}), {1.0f, -1.0f});

        float first_loss = 0.0f;
        float last_loss = 0.0f;

        for (int step = 0; step < 60; ++step)
        {
            Variable w1 = Variable::leaf(w1_value);
            Variable w2 = Variable::leaf(w2_value);
            const Variable x = Variable::leaf(x_value, false);
            const Variable minus_target = Variable::leaf(veda::ops::mul(target, -1.0f), false);

            Variable h = silu(veda::autograd::matmul_nt(x, w1));
            Variable y = veda::autograd::matmul_nt(h, w2);
            Variable error = veda::autograd::add(y, minus_target);
            Variable loss = sum(mul(error, error));

            loss.backward();

            if (step == 0)
            {
                first_loss = loss.value().at({});
            }
            last_loss = loss.value().at({});

            // plain gradient descent
            for (size_t i = 0; i < w1_value.numel(); ++i)
            {
                w1_value.data()[i] -= 0.05f * w1.grad().data()[i];
            }
            for (size_t i = 0; i < w2_value.numel(); ++i)
            {
                w2_value.data()[i] -= 0.05f * w2.grad().data()[i];
            }
        }

        CHECK(first_loss > 0.1f);
        CHECK(last_loss < first_loss * 0.05f);   // the loss fell by more than twenty times
        CHECK(last_loss >= 0.0f);
    }

    return VEDA_TEST_SUMMARY("AutogradNonLinearTest");
}
