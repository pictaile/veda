// E6.S3.T5 — nn::RMSNorm

#include "Normalize.h"
#include "RMSNorm.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <stdexcept>
#include <string>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::nn::RMSNorm;
using veda::ops::rms_normalize;

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

Tensor ones(size_t n)
{
    Tensor t{Shape({n})};
    for (size_t i = 0; i < n; ++i)
    {
        t.data()[i] = 1.0f;
    }
    return t;
}

const float eps = 1e-6f;
} // namespace

int main()
{
    // *** the acceptance criterion E2.S4.T11 set: gamma = 1 reproduces the bare op EXACTLY ***
    // Not approximately — multiplying a finite float by 1.0 is exact in IEEE 754, so if the layer
    // did anything other than scale, this fails.
    {
        Tensor x{Shape({2, 4})};
        const std::vector<float> values = {3.0f, 4.0f, -1.5f, 0.25f, 100.0f, -2.0f, 0.0f, 7.5f};
        for (size_t i = 0; i < values.size(); ++i)
        {
            x.data()[i] = values[i];
        }

        const Tensor from_layer = RMSNorm(ones(4), eps).forward(x);
        const Tensor from_op = rms_normalize(x, eps);

        CHECK_EQ(from_layer.shape(), from_op.shape());
        bool identical = true;
        for (size_t t = 0; t < 2; ++t)
        {
            for (size_t d = 0; d < 4; ++d)
            {
                if (from_layer.at({t, d}) != from_op.at({t, d}))
                {
                    identical = false;
                }
            }
        }
        CHECK(identical);
    }

    // the hand-computed case: x = [3, 4], rms = 3.5355, gamma = [2, 0.5]
    {
        const RMSNorm norm(filled(Shape({2}), {2.0f, 0.5f}), eps);
        const Tensor out = norm.forward(filled(Shape({2}), {3.0f, 4.0f}));

        CHECK_EQ(out.shape(), Shape({2}));
        CHECK_NEAR(out.at({0}), 1.69706, 1e-4);   // 0.84853 * 2
        CHECK_NEAR(out.at({1}), 0.56569, 1e-4);   // 1.13137 * 0.5
        CHECK(out.is_contiguous());

        CHECK_EQ(norm.hidden_size(), size_t{2});
        CHECK_NEAR(norm.eps(), 1e-6, 1e-12);
    }

    // gamma is per channel, never per position: every token meets the same vector
    {
        const RMSNorm norm(filled(Shape({2}), {2.0f, 0.5f}), eps);
        // the second row is twice the first, so both normalise to the same thing
        const Tensor out = norm.forward(filled(Shape({2, 2}), {3, 4, 6, 8}));

        CHECK_NEAR(out.at({0, 0}), 1.69706, 1e-4);
        CHECK_NEAR(out.at({0, 1}), 0.56569, 1e-4);
        // Scale invariance is exact in mathematics and not in fp32: sqrt(12.5) and sqrt(50)/2
        // differ in the last bits, so the two rows agree to a tolerance, not to a bit. Compare
        // with the gamma = 1 case above, which IS exact — multiplying by 1.0 introduces nothing.
        CHECK_NEAR(out.at({1, 0}), out.at({0, 0}), 1e-6);
        CHECK_NEAR(out.at({1, 1}), out.at({0, 1}), 1e-6);
    }

    // [B, T, D] — the shape the model actually uses
    {
        const size_t D = 4;
        const RMSNorm norm(filled(Shape({D}), {1.0f, 2.0f, 0.5f, 1.0f}), eps);

        Tensor x{Shape({1, 3, D})};
        for (size_t i = 0; i < x.numel(); ++i)
        {
            x.data()[i] = static_cast<float>(i + 1);
        }
        const Tensor out = norm.forward(x);
        CHECK_EQ(out.shape(), Shape({1, 3, D}));

        // each token lane still normalises independently before gamma is applied
        for (size_t t = 0; t < 3; ++t)
        {
            const Tensor bare = rms_normalize(x, eps);
            CHECK_NEAR(out.at({0, t, 1}), bare.at({0, t, 1}) * 2.0f, 1e-5);
            CHECK_NEAR(out.at({0, t, 2}), bare.at({0, t, 2}) * 0.5f, 1e-5);
        }

        // and rank 1 works: a single vector
        CHECK_EQ(norm.forward(Tensor{Shape({D})}).shape(), Shape({D}));
    }

    // a gamma of zeros switches channels off; a negative gamma flips the sign — both are legal
    // and both occur in trained models
    {
        const RMSNorm norm(filled(Shape({3}), {0.0f, -1.0f, 1.0f}), eps);
        const Tensor out = norm.forward(filled(Shape({3}), {1.0f, 1.0f, 1.0f}));

        CHECK_EQ(out.at({0}), 0.0f);
        CHECK_NEAR(out.at({1}), -1.0, 1e-5);   // a constant vector normalises to ones
        CHECK_NEAR(out.at({2}), 1.0, 1e-5);
    }

    // AD2: gamma is shared, not copied, and neither it nor the input is modified
    {
        const Tensor gamma = filled(Shape({2}), {2.0f, 0.5f});
        const RMSNorm norm(gamma, eps);

        CHECK(norm.gamma().storage() == gamma.storage());
        CHECK_EQ(gamma.storage().use_count(), long{2});

        const Tensor x = filled(Shape({2}), {3.0f, 4.0f});
        const Tensor out = norm.forward(x);
        CHECK(out.storage() != gamma.storage());
        CHECK(out.storage() != x.storage());
        CHECK_EQ(gamma.at({0}), 2.0f);
        CHECK_EQ(x.at({0}), 3.0f);
    }

    // non-contiguous input and non-contiguous gamma
    {
        const Tensor source = filled(Shape({2, 2}), {3, 6, 4, 8});
        const Tensor transposed = source.transpose(0, 1);   // rows become [3,4] and [6,8]
        CHECK(!transposed.is_contiguous());

        const RMSNorm norm(ones(2), eps);
        const Tensor out = norm.forward(transposed);
        CHECK_NEAR(out.at({0, 0}), 0.84853, 1e-4);
        CHECK_NEAR(out.at({1, 1}), 1.13137, 1e-4);

        // a gamma that is a slice of a larger buffer, as a loaded one may be
        Tensor buffer{Shape({5})};
        for (size_t i = 0; i < 5; ++i)
        {
            buffer.data()[i] = static_cast<float>(i);
        }
        const RMSNorm sliced(buffer.slice(0, 2, 2), eps);   // gamma = [2, 3]
        CHECK_EQ(sliced.hidden_size(), size_t{2});
        const Tensor scaled = sliced.forward(filled(Shape({2}), {1.0f, 1.0f}));
        CHECK_NEAR(scaled.at({0}), 2.0, 1e-5);
        CHECK_NEAR(scaled.at({1}), 3.0, 1e-5);
    }

    // refusals
    {
        const RMSNorm norm(ones(4), eps);

        CHECK_THROWS_AS(norm.forward(Tensor{Shape({2, 3})}), std::invalid_argument);
        CHECK_THROWS_AS(norm.forward(Tensor{Shape({})}), std::invalid_argument);
        try
        {
            (void)norm.forward(Tensor{Shape({1, 3})});
        }
        catch (const std::invalid_argument& error)
        {
            const std::string message = error.what();
            CHECK(message.find("(1, 3)") != std::string::npos);
            CHECK(message.find("gamma length 4") != std::string::npos);
        }

        CHECK_THROWS_AS(RMSNorm(Tensor{Shape({2, 2})}, eps), std::invalid_argument);
        CHECK_THROWS_AS(RMSNorm(Tensor{Shape({})}, eps), std::invalid_argument);
        CHECK_THROWS_AS(RMSNorm(ones(4), 0.0f), std::invalid_argument);
        CHECK_THROWS_AS(RMSNorm(ones(4), -1e-6f), std::invalid_argument);
    }

    // edge cases
    {
        // D = 1: a lane of one element normalises to +-1, then meets gamma
        const RMSNorm single(filled(Shape({1}), {3.0f}), eps);
        CHECK_NEAR(single.forward(filled(Shape({1}), {7.0f})).at({0}), 3.0, 1e-4);
        CHECK_NEAR(single.forward(filled(Shape({1}), {-7.0f})).at({0}), -3.0, 1e-4);

        // an all-zero input with eps > 0 gives zeros, not NaN
        const RMSNorm norm(ones(3), eps);
        const Tensor zeros = norm.forward(Tensor{Shape({3})});
        CHECK_EQ(zeros.at({0}), 0.0f);
        CHECK_EQ(zeros.at({2}), 0.0f);

        // an empty sequence
        CHECK_EQ(norm.forward(Tensor{Shape({0, 3})}).shape(), Shape({0, 3}));
    }

    return VEDA_TEST_SUMMARY("RMSNormLayerTest");
}
