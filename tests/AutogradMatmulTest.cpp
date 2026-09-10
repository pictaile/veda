// E14.S2.T3 — the matmul gradients

#include "GradientCheck.h"
#include "Matmul.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"
#include "Variable.h"

#include <random>
#include <vector>

using veda::autograd::check_gradient;
using veda::autograd::matmul;
using veda::autograd::matmul_nt;
using veda::autograd::mul;
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
    std::uniform_real_distribution<float> values(-1.0f, 1.0f);
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
    // --- the hand-computed 2x2 -------------------------------------------------------------------
    //   A = [[1,2],[3,4]]   B = [[5,6],[7,8]]   G = all ones
    //   dL/dA = G . Bt = [[11,15],[11,15]]
    //   dL/dB = At . G  = [[4,4],[6,6]]
    {
        Variable a = Variable::leaf(filled(Shape({2, 2}), {1, 2, 3, 4}));
        Variable b = Variable::leaf(filled(Shape({2, 2}), {5, 6, 7, 8}));

        Variable c = matmul(a, b);
        CHECK_EQ(c.value().at({0, 0}), 19.0f);   // the E2.S2.T4 numbers
        CHECK_EQ(c.value().at({1, 1}), 50.0f);

        sum(c).backward();   // G is all ones

        CHECK_NEAR(a.grad().at({0, 0}), 11.0, 1e-4);
        CHECK_NEAR(a.grad().at({0, 1}), 15.0, 1e-4);
        CHECK_NEAR(a.grad().at({1, 0}), 11.0, 1e-4);
        CHECK_NEAR(a.grad().at({1, 1}), 15.0, 1e-4);

        CHECK_NEAR(b.grad().at({0, 0}), 4.0, 1e-4);
        CHECK_NEAR(b.grad().at({0, 1}), 4.0, 1e-4);
        CHECK_NEAR(b.grad().at({1, 0}), 6.0, 1e-4);
        CHECK_NEAR(b.grad().at({1, 1}), 6.0, 1e-4);

        // the gradient has the shape of its input, always
        CHECK_EQ(a.grad().shape(), Shape({2, 2}));
        CHECK_EQ(b.grad().shape(), Shape({2, 2}));
    }

    // *** a non-square case: a misplaced transpose gives the right shape when M == N ***
    {
        std::mt19937 engine(20260910);
        const Tensor a_value = random_tensor(Shape({2, 3}), engine);   // M=2, K=3
        const Tensor b_value = random_tensor(Shape({3, 4}), engine);   // K=3, N=4

        Variable a = Variable::leaf(a_value);
        Variable b = Variable::leaf(b_value);
        Variable loss = sum(matmul(a, b));
        loss.backward();

        CHECK_EQ(a.grad().shape(), Shape({2, 3}));
        CHECK_EQ(b.grad().shape(), Shape({3, 4}));

        // dL/dA[i,k] = sum over j of B[k,j], since every G is 1
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t k = 0; k < 3; ++k)
            {
                float expected = 0.0f;
                for (size_t j = 0; j < 4; ++j)
                {
                    expected += b_value.at({k, j});
                }
                CHECK_NEAR(a.grad().at({i, k}), expected, 1e-4);
            }
        }

        // dL/dB[k,j] = sum over i of A[i,k]
        for (size_t k = 0; k < 3; ++k)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                float expected = 0.0f;
                for (size_t i = 0; i < 2; ++i)
                {
                    expected += a_value.at({i, k});
                }
                CHECK_NEAR(b.grad().at({k, j}), expected, 1e-4);
            }
        }
    }

    // *** finite differences, with respect to each operand separately ***
    {
        std::mt19937 engine(31415);
        const Tensor a_value = random_tensor(Shape({3, 4}), engine);
        const Tensor b_value = random_tensor(Shape({4, 2}), engine);
        const Tensor weights = random_tensor(Shape({3, 2}), engine);   // to make the loss non-trivial

        const Variable fixed_b = Variable::leaf(b_value, false);
        const Variable fixed_a = Variable::leaf(a_value, false);
        const Variable scale = Variable::leaf(weights, false);

        // d/dA
        CHECK(check_gradient(
                  [&](const Variable& v) { return sum(mul(matmul(v, fixed_b), scale)); }, a_value)
                  .ok);
        // d/dB
        CHECK(check_gradient(
                  [&](const Variable& v) { return sum(mul(matmul(fixed_a, v), scale)); }, b_value)
                  .ok);
    }

    // --- matmul_nt, the mirror ---------------------------------------------------------------------
    {
        // A [2,3] x B [4,3] -> C [2,4]
        std::mt19937 engine(2718);
        const Tensor a_value = random_tensor(Shape({2, 3}), engine);
        const Tensor b_value = random_tensor(Shape({4, 3}), engine);
        const Tensor scale_value = random_tensor(Shape({2, 4}), engine);

        Variable a = Variable::leaf(a_value);
        Variable b = Variable::leaf(b_value);
        const Variable scale = Variable::leaf(scale_value, false);

        Variable loss = sum(mul(matmul_nt(a, b), scale));
        loss.backward();

        CHECK_EQ(a.grad().shape(), Shape({2, 3}));
        CHECK_EQ(b.grad().shape(), Shape({4, 3}));

        const Variable fixed_b = Variable::leaf(b_value, false);
        const Variable fixed_a = Variable::leaf(a_value, false);
        CHECK(check_gradient(
                  [&](const Variable& v) { return sum(mul(matmul_nt(v, fixed_b), scale)); },
                  a_value)
                  .ok);
        CHECK(check_gradient(
                  [&](const Variable& v) { return sum(mul(matmul_nt(fixed_a, v), scale)); },
                  b_value)
                  .ok);

        // the mirror stated as an identity: matmul_nt(A, B) == matmul(A, transpose(B)), so their
        // gradients with respect to A agree
        Variable a2 = Variable::leaf(a_value);
        const Variable b_transposed = Variable::leaf(veda::core::contiguous(b_value.transpose(0, 1)),
                                                      false);
        sum(mul(matmul(a2, b_transposed), scale)).backward();

        for (size_t i = 0; i < a.grad().numel(); ++i)
        {
            CHECK_NEAR(a.grad().data()[i], a2.grad().data()[i], 1e-4);
        }
    }

    // *** the batched case: a weight meets activations, and its gradient is summed over the batch ***
    {
        const size_t B = 2;
        const size_t T = 3;
        const size_t in = 4;
        const size_t out = 5;

        std::mt19937 engine(161803);
        const Tensor activations = random_tensor(Shape({B, T, in}), engine);
        const Tensor weight_value = random_tensor(Shape({out, in}), engine);
        const Tensor scale_value = random_tensor(Shape({B, T, out}), engine);

        Variable x = Variable::leaf(activations);
        Variable w = Variable::leaf(weight_value);
        const Variable scale = Variable::leaf(scale_value, false);

        Variable y = matmul_nt(x, w);   // [B, T, out] — exactly nn::Linear
        CHECK_EQ(y.value().shape(), Shape({B, T, out}));

        sum(mul(y, scale)).backward();

        // the parameter gradient has the parameter's shape, summed over B and T
        CHECK_EQ(w.grad().shape(), Shape({out, in}));
        CHECK_EQ(x.grad().shape(), Shape({B, T, in}));

        // and it is the sum, not one batch element's worth: check one entry by hand
        float expected = 0.0f;
        for (size_t b = 0; b < B; ++b)
        {
            for (size_t t = 0; t < T; ++t)
            {
                expected += scale_value.at({b, t, 1}) * activations.at({b, t, 2});
            }
        }
        CHECK_NEAR(w.grad().at({1, 2}), expected, 1e-4);

        // finite differences on the weight — the check that catches a gradient of the right shape
        // that is quietly B*T times too large
        const Variable fixed_x = Variable::leaf(activations, false);
        CHECK(check_gradient(
                  [&](const Variable& v) { return sum(mul(matmul_nt(fixed_x, v), scale)); },
                  weight_value, 1e-2f, 3e-2)
                  .ok);
    }

    // batched on both sides
    {
        std::mt19937 engine(1414);
        const Tensor a_value = random_tensor(Shape({2, 3, 4}), engine);
        const Tensor b_value = random_tensor(Shape({2, 4, 3}), engine);
        const Tensor scale_value = random_tensor(Shape({2, 3, 3}), engine);
        const Variable scale = Variable::leaf(scale_value, false);

        Variable a = Variable::leaf(a_value);
        Variable b = Variable::leaf(b_value);
        sum(mul(matmul(a, b), scale)).backward();

        CHECK_EQ(a.grad().shape(), Shape({2, 3, 4}));
        CHECK_EQ(b.grad().shape(), Shape({2, 4, 3}));

        const Variable fixed_b = Variable::leaf(b_value, false);
        CHECK(check_gradient(
                  [&](const Variable& v) { return sum(mul(matmul(v, fixed_b), scale)); }, a_value)
                  .ok);
    }

    // no requires_grad means no tape
    {
        Variable a = Variable::leaf(Tensor{Shape({2, 2})}, false);
        Variable b = Variable::leaf(Tensor{Shape({2, 2})}, false);
        Variable c = matmul(a, b);
        CHECK(!c.requires_grad());
        CHECK(c.node()->parents.empty());
    }

    // the cost of training, as arithmetic: the backward is two matmuls where the forward was one
    {
        const size_t forward_matmuls = 1;
        const size_t backward_matmuls = 2;
        CHECK_EQ(forward_matmuls + backward_matmuls, size_t{3});   // ~3x a forward pass per step
    }

    return VEDA_TEST_SUMMARY("AutogradMatmulTest");
}
