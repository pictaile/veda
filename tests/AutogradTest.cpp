// E14.S1.T1 (reverse mode) and T2 (the tape and gradient checking)

#include "GradientCheck.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"
#include "Elementwise.h"
#include "Variable.h"

#include <memory>

#include <stdexcept>
#include <string>
#include <vector>

using veda::autograd::add;
using veda::autograd::check_gradient;
using veda::autograd::GradientCheck;
using veda::autograd::mul;
using veda::autograd::sum;
using veda::autograd::Variable;
using veda::core::Shape;
using veda::core::Tensor;

namespace
{
Tensor scalar(float value)
{
    Tensor t{Shape({})};
    t.data()[0] = value;
    return t;
}

Tensor filled(const Shape& shape, const std::vector<float>& values)
{
    Tensor t{shape};
    for (size_t i = 0; i < values.size(); ++i)
    {
        t.data()[i] = values[i];
    }
    return t;
}
} // namespace

int main()
{
    // --- the worked example: x = 3, w = 2, h = xw, y = h^2, L = y ------------------------------
    {
        Variable x = Variable::leaf(scalar(3.0f));
        Variable w = Variable::leaf(scalar(2.0f));

        Variable h = mul(x, w);
        Variable y = mul(h, h);
        Variable loss = sum(y);

        CHECK_EQ(h.value().at({}), 6.0f);
        CHECK_EQ(loss.value().at({}), 36.0f);

        loss.backward();

        CHECK_NEAR(x.grad().at({}), 24.0, 1e-4);   // 2*h*w
        CHECK_NEAR(w.grad().at({}), 36.0, 1e-4);   // 2*h*x
        CHECK_NEAR(h.grad().at({}), 12.0, 1e-4);   // 2*h
    }

    // *** the case that separates accumulation from assignment ***
    {
        Variable x = Variable::leaf(scalar(3.0f));
        sum(add(x, x)).backward();
        CHECK_NEAR(x.grad().at({}), 2.0, 1e-5);    // not 1

        Variable y = Variable::leaf(scalar(4.0f));
        sum(mul(y, y)).backward();
        CHECK_NEAR(y.grad().at({}), 8.0, 1e-5);    // 2x, not x
    }

    // zero_grad clears, and a second backward accumulates again
    {
        Variable x = Variable::leaf(scalar(2.0f));
        Variable loss = sum(mul(x, x));

        loss.backward();
        CHECK_NEAR(x.grad().at({}), 4.0, 1e-5);

        loss.backward();
        CHECK_NEAR(x.grad().at({}), 8.0, 1e-5);    // accumulated, as it should

        loss.zero_grad();
        CHECK_EQ(x.grad().at({}), 0.0f);
        loss.backward();
        CHECK_NEAR(x.grad().at({}), 4.0, 1e-5);
    }

    // *** no requires_grad anywhere means no tape at all — inference pays nothing ***
    {
        Variable a = Variable::leaf(filled(Shape({3}), {1, 2, 3}), false);
        Variable b = Variable::leaf(filled(Shape({3}), {4, 5, 6}), false);

        Variable c = mul(add(a, b), 2.0f);
        CHECK(!c.requires_grad());
        CHECK(c.node()->parents.empty());
        CHECK(!c.node()->backward);
        CHECK_EQ(c.value().at({0}), 10.0f);

        // and one requiring a gradient is enough to record
        Variable d = Variable::leaf(filled(Shape({3}), {1, 1, 1}), true);
        Variable e = mul(a, d);
        CHECK(e.requires_grad());
        CHECK_EQ(e.node()->parents.size(), size_t{2});
    }

    // gradients have the shape of their value, always
    {
        Variable x = Variable::leaf(Tensor{Shape({2, 3})});
        CHECK_EQ(x.grad().shape(), Shape({2, 3}));

        Variable loss = sum(mul(x, 3.0f));
        loss.backward();
        CHECK_EQ(x.grad().shape(), Shape({2, 3}));
        for (size_t i = 0; i < 6; ++i)
        {
            CHECK_NEAR(x.grad().data()[i], 3.0, 1e-5);
        }
    }

    // *** the backward of a broadcast is a sum over the broadcast axes ***
    {
        Variable wide = Variable::leaf(filled(Shape({2, 3}), {1, 2, 3, 4, 5, 6}));
        Variable narrow = Variable::leaf(filled(Shape({3}), {1, 1, 1}));

        sum(mul(wide, narrow)).backward();

        // the [3] input contributed to two rows, so its gradient is the column sums of `wide`
        CHECK_EQ(narrow.grad().shape(), Shape({3}));
        CHECK_NEAR(narrow.grad().at({0}), 1.0 + 4.0, 1e-5);
        CHECK_NEAR(narrow.grad().at({1}), 2.0 + 5.0, 1e-5);
        CHECK_NEAR(narrow.grad().at({2}), 3.0 + 6.0, 1e-5);

        // and the wide one gets the narrow one's values, broadcast back
        CHECK_EQ(wide.grad().shape(), Shape({2, 3}));
        CHECK_NEAR(wide.grad().at({1, 2}), 1.0, 1e-5);
    }

    // backward starts at a scalar
    {
        Variable x = Variable::leaf(Tensor{Shape({2, 2})});
        Variable y = mul(x, 2.0f);
        CHECK_THROWS_AS(y.backward(), std::runtime_error);
    }

    // --- the instrument, checked against the analytic gradients it will police -------------------
    {
        const Tensor point = filled(Shape({4}), {0.5f, -1.5f, 2.0f, 0.25f});

        // sum(x)          -> 1
        CHECK(check_gradient([](const Variable& v) { return sum(v); }, point).ok);
        // sum(x * x)      -> 2x
        CHECK(check_gradient([](const Variable& v) { return sum(mul(v, v)); }, point).ok);
        // sum(3x)         -> 3
        CHECK(check_gradient([](const Variable& v) { return sum(mul(v, 3.0f)); }, point).ok);
        // sum(x + x)      -> 2
        CHECK(check_gradient([](const Variable& v) { return sum(add(v, v)); }, point).ok);
        // sum(x * x * x)  -> 3x^2
        CHECK(check_gradient([](const Variable& v) { return sum(mul(mul(v, v), v)); }, point,
                             1e-2f, 5e-2)
                  .ok);

        // on a matrix, and with a broadcast inside
        const Tensor matrix = filled(Shape({2, 3}), {1, -2, 3, 0.5f, 2, -1});
        CHECK(check_gradient([](const Variable& v) { return sum(mul(v, v)); }, matrix).ok);
    }

    // *** the instrument catches a wrong gradient — proven, not assumed ***
    {
        // A deliberately broken op: the value is x*x but the gradient claims x rather than 2x.
        const auto broken = [](const Variable& v) {
            Tensor value = veda::ops::mul(v.value(), v.value());
            auto input = v.node();
            auto node = std::make_shared<veda::autograd::Node>(std::move(value));
            node->requires_grad = true;
            node->parents = {input};
            node->backward = [input](const Tensor& g) {
                // wrong by a factor of two — exactly the bug that still trains, just worse
                for (size_t i = 0; i < input->grad.numel(); ++i)
                {
                    input->grad.data()[i] += g.data()[i] * input->value.data()[i];
                }
            };
            return sum(Variable(std::move(node)));
        };

        const GradientCheck result =
            check_gradient(broken, filled(Shape({3}), {1.0f, 2.0f, 3.0f}));

        CHECK(!result.ok);
        CHECK(result.worst_relative > 0.4);
        CHECK(result.message.find("gradient mismatch") != std::string::npos);
        CHECK(result.message.find("index") != std::string::npos);
    }

    // a function that does not return a scalar is refused
    {
        const GradientCheck result =
            check_gradient([](const Variable& v) { return mul(v, 2.0f); }, filled(Shape({2}), {1, 2}));
        CHECK(!result.ok);
        CHECK(result.message.find("must return a scalar") != std::string::npos);
    }

    // --- why reverse mode: the sweep counts ------------------------------------------------------
    {
        const size_t parameters = 596'000'000;
        const size_t outputs = 1;             // the loss is a scalar

        CHECK_EQ(outputs, size_t{1});
        CHECK(parameters / outputs > 500'000'000);   // forward mode would need one sweep per input
    }

    return VEDA_TEST_SUMMARY("AutogradTest");
}
