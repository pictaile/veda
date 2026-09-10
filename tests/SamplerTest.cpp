// E11.S1.T1 (why the sampler is separate) and T2 (GreedySampler)

#include "Sampling.h"
#include "Shape.h"
#include "Softmax.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::generate::GreedySampler;
using veda::generate::Sampler;

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
} // namespace

int main()
{
    GreedySampler greedy;

    // the worked example
    {
        CHECK_EQ(greedy.sample(filled(Shape({4}), {1.0f, 3.0f, 2.0f, 0.5f})), 1);
        CHECK_EQ(greedy.sample(filled(Shape({3}), {-5.0f, -1.0f, -9.0f})), 1);   // negatives are fine
        CHECK_EQ(greedy.sample(filled(Shape({1}), {42.0f})), 0);
    }

    // *** the tie rule: the lower id wins, and it is stated rather than incidental ***
    {
        CHECK_EQ(greedy.sample(filled(Shape({2}), {2.0f, 2.0f})), 0);
        CHECK_EQ(greedy.sample(filled(Shape({4}), {1.0f, 5.0f, 5.0f, 5.0f})), 1);
        CHECK_EQ(greedy.sample(filled(Shape({3}), {0.0f, 0.0f, 0.0f})), 0);
    }

    // both shapes the generation loop can hand over
    {
        CHECK_EQ(greedy.sample(filled(Shape({1, 4}), {1.0f, 3.0f, 2.0f, 0.5f})), 1);
        CHECK_THROWS_AS(greedy.sample(filled(Shape({2, 2}), {1, 2, 3, 4})), std::invalid_argument);
        CHECK_THROWS_AS(greedy.sample(filled(Shape({4, 1}), {1, 2, 3, 4})), std::invalid_argument);
        CHECK_THROWS_AS(greedy.sample(Tensor{Shape({1, 1, 4})}), std::invalid_argument);
        CHECK_THROWS_AS(greedy.sample(Tensor{Shape({0})}), std::invalid_argument);
    }

    // *** a NaN is a bug upstream, not something to sample around ***
    // Every comparison with NaN is false, so a naive argmax would silently return index 0.
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        CHECK_THROWS_AS(greedy.sample(filled(Shape({3}), {1.0f, nan, 2.0f})), std::runtime_error);
        try
        {
            (void)greedy.sample(filled(Shape({3}), {1.0f, 2.0f, nan}));
        }
        catch (const std::runtime_error& error)
        {
            CHECK(std::string(error.what()).find("logit 2") != std::string::npos);
        }

        // infinities are legal: -inf is what a mask produces, and +inf is a very confident model
        CHECK_EQ(greedy.sample(filled(Shape({3}),
                                      {-std::numeric_limits<float>::infinity(), 1.0f, 2.0f})),
                 2);
        CHECK_EQ(greedy.sample(filled(Shape({2}), {std::numeric_limits<float>::infinity(), 5.0f})),
                 0);
    }

    // *** greedy is argmax, and argmax needs no softmax: the two agree on random logits ***
    {
        std::mt19937 engine(20260908);
        std::normal_distribution<float> values(0.0f, 5.0f);

        for (int trial = 0; trial < 50; ++trial)
        {
            const size_t vocabulary = 1 + static_cast<size_t>(engine() % 64);
            Tensor logits{Shape({vocabulary})};
            for (size_t v = 0; v < vocabulary; ++v)
            {
                logits.data()[v] = values(engine);
            }

            // a hand-written argmax
            size_t expected = 0;
            for (size_t v = 1; v < vocabulary; ++v)
            {
                if (logits.at({v}) > logits.at({expected}))
                {
                    expected = v;
                }
            }
            CHECK_EQ(greedy.sample(logits), static_cast<int32_t>(expected));

            // and the argmax of the softmax is the same id — which is why normalising here would
            // spend V exponentials to change nothing
            const Tensor probabilities = veda::ops::softmax(logits, 0);
            size_t after_softmax = 0;
            for (size_t v = 1; v < vocabulary; ++v)
            {
                if (probabilities.at({v}) > probabilities.at({after_softmax}))
                {
                    after_softmax = v;
                }
            }
            CHECK_EQ(after_softmax, expected);
        }
    }

    // deterministic: the same logits give the same id, every time
    {
        const Tensor logits = filled(Shape({5}), {0.1f, 0.9f, 0.3f, 0.9f, 0.2f});
        const int32_t first = greedy.sample(logits);
        for (int i = 0; i < 10; ++i)
        {
            CHECK_EQ(greedy.sample(logits), first);
        }
        CHECK_EQ(first, 1);   // the tie between 1 and 3 goes to the lower id
    }

    // the interface is what later samplers implement — a reference held by base pointer works
    {
        GreedySampler concrete;
        Sampler& sampler = concrete;
        CHECK_EQ(sampler.sample(filled(Shape({3}), {1.0f, 7.0f, 3.0f})), 1);
    }

    // the input is not modified
    {
        const Tensor logits = filled(Shape({3}), {1.0f, 2.0f, 3.0f});
        (void)greedy.sample(logits);
        CHECK_EQ(logits.at({0}), 1.0f);
        CHECK_EQ(logits.at({2}), 3.0f);
    }

    return VEDA_TEST_SUMMARY("SamplerTest");
}
