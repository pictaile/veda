// E15.S1.T1 — cross-entropy, and why its backward is a subtraction

#include "GradientCheck.h"
#include "Loss.h"
#include "Shape.h"
#include "Softmax.h"
#include "TestSupport.h"
#include "Tensor.h"
#include "Variable.h"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using veda::autograd::check_gradient;
using veda::autograd::cross_entropy;
using veda::autograd::Variable;
using veda::core::Shape;
using veda::core::Tensor;
using veda::ops::log_softmax;
using veda::ops::softmax;

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
    std::uniform_real_distribution<float> values(-2.0f, 2.0f);
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
    std::mt19937 engine(20260911);

    // --- log_softmax against the tested softmax --------------------------------------------------
    {
        const Tensor x = filled(Shape({3}), {1.0f, 2.0f, 3.0f});
        const Tensor logp = log_softmax(x, 0);

        // m = 3, sum exp(x - m) = 1.5032, log = 0.4076
        CHECK_NEAR(logp.at({0}), -2.4076, 1e-3);
        CHECK_NEAR(logp.at({1}), -1.4076, 1e-3);
        CHECK_NEAR(logp.at({2}), -0.4076, 1e-3);

        const Tensor p = softmax(x, 0);
        for (size_t i = 0; i < 3; ++i)
        {
            CHECK_NEAR(std::exp(logp.at({i})), p.at({i}), 1e-6);
        }
    }

    // *** the case that decides the design: softmax is safe at the top and unsafe at the bottom ***
    {
        const Tensor x = filled(Shape({2}), {0.0f, 200.0f});

        const Tensor p = softmax(x, 0);
        CHECK_EQ(p.at({0}), 0.0f);                     // underflowed to exactly zero
        CHECK(!std::isfinite(std::log(p.at({0}))));    // and log of it is not a number to train on

        const Tensor logp = log_softmax(x, 0);
        CHECK(std::isfinite(logp.at({0})));
        CHECK_NEAR(logp.at({0}), -200.0, 1e-2);        // large, correct, differentiable
        CHECK_NEAR(logp.at({1}), 0.0, 1e-2);
    }

    // rows of a matrix, and along different axes; a fully masked lane stays -inf, not NaN
    {
        std::mt19937 local(7);
        const Tensor m = random_tensor(Shape({3, 4}), local);
        for (size_t dim = 0; dim < 2; ++dim)
        {
            const Tensor logp = log_softmax(m, dim);
            const Tensor p = softmax(m, dim);
            for (size_t r = 0; r < 3; ++r)
            {
                for (size_t c = 0; c < 4; ++c)
                {
                    CHECK_NEAR(std::exp(logp.at({r, c})), p.at({r, c}), 1e-6);
                }
            }
        }

        const float minus_infinity = -std::numeric_limits<float>::infinity();
        const Tensor masked = filled(Shape({3}), {minus_infinity, 0.0f, minus_infinity});
        const Tensor masked_logp = log_softmax(masked, 0);
        CHECK(masked_logp.at({0}) == minus_infinity);
        CHECK_NEAR(masked_logp.at({1}), 0.0, 1e-6);

        const Tensor all_masked =
            filled(Shape({2}), {minus_infinity, minus_infinity});
        const Tensor all_logp = log_softmax(all_masked, 0);
        CHECK(all_logp.at({0}) == minus_infinity);     // -inf, never NaN
        CHECK(!std::isnan(all_logp.at({1})));

        CHECK_THROWS_AS(log_softmax(m, 2), std::invalid_argument);
    }

    // --- the loss, and the gradient that is a subtraction ------------------------------------------
    {
        Variable logits = Variable::leaf(filled(Shape({1, 3}), {1.0f, 2.0f, 3.0f}));
        Variable loss = cross_entropy(logits, {2});

        CHECK_EQ(loss.value().shape(), Shape({}));
        CHECK_NEAR(loss.value().at({}), 0.4076, 1e-3);

        loss.backward();

        // *** dL/dx = p - one_hot ***
        CHECK_NEAR(logits.grad().at({0, 0}), 0.0900, 1e-3);
        CHECK_NEAR(logits.grad().at({0, 1}), 0.2447, 1e-3);
        CHECK_NEAR(logits.grad().at({0, 2}), -0.3348, 1e-3);
        CHECK_EQ(logits.grad().shape(), Shape({1, 3}));

        float total = 0.0f;
        for (size_t i = 0; i < 3; ++i)
        {
            total += logits.grad().at({0, i});
        }
        CHECK_NEAR(total, 0.0, 1e-5);   // sum p = 1, minus one one-hot
    }

    // the confident-and-wrong case: the gradient on the true token approaches -1
    {
        Variable logits = Variable::leaf(filled(Shape({1, 3}), {1.0f, 2.0f, 3.0f}));
        Variable loss = cross_entropy(logits, {0});
        CHECK_NEAR(loss.value().at({}), 2.4076, 1e-3);

        loss.backward();
        CHECK_NEAR(logits.grad().at({0, 0}), -0.9100, 1e-3);
        CHECK_NEAR(logits.grad().at({0, 1}), 0.2447, 1e-3);
        CHECK_NEAR(logits.grad().at({0, 2}), 0.6652, 1e-3);
    }

    // the two reference points of any training run
    {
        // a confident correct prediction: loss near zero, and a gradient near zero with it
        Variable confident = Variable::leaf(filled(Shape({1, 4}), {0.0f, 0.0f, 20.0f, 0.0f}));
        Variable loss = cross_entropy(confident, {2});
        CHECK(loss.value().at({}) < 1e-6f);
        loss.backward();
        for (size_t i = 0; i < 4; ++i)
        {
            CHECK(std::fabs(confident.grad().at({0, i})) < 1e-5f);
        }

        // *** uniform logits over V give log V — where an untrained model sits ***
        const size_t vocabulary = 1000;
        const Variable uniform = Variable::leaf(Tensor{Shape({1, vocabulary})}, false);
        const Variable uniform_loss = cross_entropy(uniform, {17});
        CHECK_NEAR(uniform_loss.value().at({}), std::log(static_cast<double>(vocabulary)), 1e-4);
        CHECK(!uniform_loss.requires_grad());   // no requires_grad -> no tape
        CHECK(uniform_loss.node()->parents.empty());
    }

    // --- the mean over lanes, and its 1/lanes in the backward -------------------------------------
    {
        const Tensor rows = filled(Shape({2, 2, 3}),
                                   {1, 2, 3, 3, 2, 1, 0, 0, 0, 5, 1, 1});
        const std::vector<int64_t> targets = {2, 0, 1, 0};

        Variable batched = Variable::leaf(rows);
        Variable loss = cross_entropy(batched, targets);

        // the same four lanes scored one at a time, averaged by hand
        double by_hand = 0.0;
        for (size_t lane = 0; lane < 4; ++lane)
        {
            const Tensor single = rows.reshape(Shape({4, 3})).slice(0, lane, 1);
            by_hand += cross_entropy(Variable::leaf(single, false), {targets[lane]})
                           .value()
                           .at({});
        }
        CHECK_NEAR(loss.value().at({}), by_hand / 4.0, 1e-5);

        loss.backward();

        // every element is (p - one_hot)/4, so the whole gradient still sums to zero
        float total = 0.0f;
        for (size_t i = 0; i < batched.grad().numel(); ++i)
        {
            total += batched.grad().data()[i];
        }
        CHECK_NEAR(total, 0.0, 1e-5);

        // and no element can exceed 1/lanes in magnitude: cross-entropy cannot make an enormous
        // gradient out of an enormous logit, which is why it is the loss used with softmax
        for (size_t i = 0; i < batched.grad().numel(); ++i)
        {
            CHECK(std::fabs(batched.grad().data()[i]) <= 1.0f / 4.0f + 1e-6f);
        }

        // the scale itself: the same lane alone must give four times this gradient
        Variable alone = Variable::leaf(veda::core::contiguous(rows.reshape(Shape({4, 3})).slice(0, 0, 1)));
        cross_entropy(alone, {targets[0]}).backward();
        CHECK_NEAR(alone.grad().at({0, 0}), batched.grad().at({0, 0, 0}) * 4.0, 1e-5);
    }

    // --- finite differences -------------------------------------------------------------------------
    {
        const Tensor point = random_tensor(Shape({1, 6}), engine);
        CHECK(check_gradient([](const Variable& v) { return cross_entropy(v, {3}); }, point).ok);

        const Tensor batch = random_tensor(Shape({3, 5}), engine);
        const std::vector<int64_t> targets = {0, 4, 2};
        CHECK(check_gradient([&](const Variable& v) { return cross_entropy(v, targets); }, batch)
                  .ok);

        const Tensor deep = random_tensor(Shape({2, 2, 4}), engine);
        const std::vector<int64_t> deep_targets = {1, 3, 0, 2};
        CHECK(check_gradient(
                  [&](const Variable& v) { return cross_entropy(v, deep_targets); }, deep)
                  .ok);
    }

    // --- refusals ------------------------------------------------------------------------------------
    {
        const Variable logits = Variable::leaf(Tensor{Shape({2, 3})}, false);

        CHECK_THROWS_AS(cross_entropy(logits, {0}), std::invalid_argument);          // one target, two lanes
        CHECK_THROWS_AS(cross_entropy(logits, {0, 1, 2}), std::invalid_argument);    // three
        CHECK_THROWS_AS(cross_entropy(logits, {0, 3}), std::invalid_argument);       // 3 >= V
        CHECK_THROWS_AS(cross_entropy(logits, {0, -1}), std::invalid_argument);      // negative
        CHECK_THROWS_AS(cross_entropy(Variable::leaf(Tensor{Shape({})}, false), {0}),
                        std::invalid_argument);

        try
        {
            (void)cross_entropy(logits, {0, 7});
        }
        catch (const std::invalid_argument& error)
        {
            const std::string message = error.what();
            CHECK(message.find("target 7") != std::string::npos);
            CHECK(message.find("[0, 3)") != std::string::npos);
        }
    }

    return VEDA_TEST_SUMMARY("CrossEntropyTest");
}
