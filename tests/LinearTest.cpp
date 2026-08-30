// E6.S2.T4 — nn::Linear

#include "Linear.h"
#include "Matmul.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
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

// W = [[1,0,2],[0,1,1]] — two output features from three input ones
Tensor weight()
{
    return filled(Shape({2, 3}), {1, 0, 2, 0, 1, 1});
}
} // namespace

int main()
{
    // the hand-computed example, with and without bias
    {
        const Linear projection(weight());
        CHECK_EQ(projection.in_features(), size_t{3});
        CHECK_EQ(projection.out_features(), size_t{2});
        CHECK(!projection.has_bias());

        const Tensor x = filled(Shape({1, 3}), {1, 2, 3});
        const Tensor y = projection.forward(x);

        CHECK_EQ(y.shape(), Shape({1, 2}));
        CHECK_EQ(y.at({0, 0}), 7.0f);   // 1*1 + 0*2 + 2*3
        CHECK_EQ(y.at({0, 1}), 5.0f);   // 0*1 + 1*2 + 1*3
        CHECK(y.is_contiguous());

        const Linear with_bias(weight(), filled(Shape({2}), {10, 20}));
        CHECK(with_bias.has_bias());
        const Tensor biased = with_bias.forward(x);
        CHECK_EQ(biased.at({0, 0}), 17.0f);
        CHECK_EQ(biased.at({0, 1}), 25.0f);

        // the input is untouched by either
        CHECK_EQ(x.at({0, 0}), 1.0f);
    }

    // *** the AD4 identity, now at the layer level ***
    // forward(x) must equal matmul(x, weight.transpose(0, 1)) — the layout convention, checked
    {
        std::mt19937 engine(20260830);
        for (const std::vector<size_t>& dims : std::vector<std::vector<size_t>>{
                 {4, 3, 2}, {1, 8, 5}, {7, 1, 3}, {2, 2, 2}})
        {
            const size_t rows = dims[0];
            const size_t in = dims[1];
            const size_t out = dims[2];

            const Tensor w = random_tensor(Shape({out, in}), engine);
            const Tensor x = random_tensor(Shape({rows, in}), engine);

            const Tensor from_layer = Linear(w).forward(x);
            const Tensor from_transpose = veda::ops::matmul(x, w.transpose(0, 1));

            CHECK_EQ(from_layer.shape(), Shape({rows, out}));
            bool identical = true;
            for (size_t i = 0; i < rows && identical; ++i)
            {
                for (size_t j = 0; j < out; ++j)
                {
                    if (from_layer.at({i, j}) != from_transpose.at({i, j}))
                    {
                        identical = false;
                        break;
                    }
                }
            }
            CHECK(identical);
        }
    }

    // leading dimensions pass through untouched, at every rank the model uses
    {
        const Linear projection(Tensor{Shape({4, 3})});
        CHECK_EQ(projection.forward(Tensor{Shape({8, 3})}).shape(), Shape({8, 4}));
        CHECK_EQ(projection.forward(Tensor{Shape({1, 8, 3})}).shape(), Shape({1, 8, 4}));
        CHECK_EQ(projection.forward(Tensor{Shape({1, 16, 8, 3})}).shape(), Shape({1, 16, 8, 4}));

        // and the bias broadcasts across all of them
        const Linear biased(Tensor{Shape({4, 3})}, filled(Shape({4}), {1, 2, 3, 4}));
        const Tensor out = biased.forward(Tensor{Shape({1, 2, 3})});
        CHECK_EQ(out.shape(), Shape({1, 2, 4}));
        CHECK_EQ(out.at({0, 0, 0}), 1.0f);   // zero input plus the bias
        CHECK_EQ(out.at({0, 1, 3}), 4.0f);
    }

    // the real projection shapes of a transformer block
    {
        const size_t D = 64;
        const size_t F = 192;
        const size_t H = 4;
        const size_t HKV = 2;
        const size_t DH = 16;

        const Tensor x{Shape({1, 8, D})};   // [B, T, D]

        CHECK_EQ(Linear(Tensor{Shape({H * DH, D})}).forward(x).shape(), Shape({1, 8, 64}));
        CHECK_EQ(Linear(Tensor{Shape({HKV * DH, D})}).forward(x).shape(), Shape({1, 8, 32}));
        CHECK_EQ(Linear(Tensor{Shape({F, D})}).forward(x).shape(), Shape({1, 8, 192}));

        const Tensor hidden{Shape({1, 8, F})};
        CHECK_EQ(Linear(Tensor{Shape({D, F})}).forward(hidden).shape(), Shape({1, 8, 64}));
    }

    // AD2: nothing is allocated or copied
    {
        const Tensor w = weight();
        const Tensor b = filled(Shape({2}), {1, 1});
        const Linear projection(w, b);

        CHECK(projection.weight().storage() == w.storage());
        CHECK_EQ(w.storage().use_count(), long{2});

        const Tensor y = projection.forward(filled(Shape({1, 3}), {1, 1, 1}));
        CHECK(y.storage() != w.storage());
        CHECK_EQ(w.at({0, 0}), 1.0f);   // the weight is unmodified
        CHECK_EQ(b.at({0}), 1.0f);
    }

    // a non-contiguous weight — a real one may be a slice of the loader's buffer
    {
        Tensor buffer{Shape({4, 3})};
        for (size_t i = 0; i < buffer.numel(); ++i)
        {
            buffer.data()[i] = static_cast<float>(i);
        }
        const Tensor slice = buffer.slice(0, 1, 2);   // rows 1..2, offset 3
        CHECK_EQ(slice.offset(), size_t{3});

        const Linear projection(slice);
        const Tensor y = projection.forward(filled(Shape({1, 3}), {1, 0, 0}));
        CHECK_EQ(y.at({0, 0}), 3.0f);   // row 1 of the buffer starts with 3
        CHECK_EQ(y.at({0, 1}), 6.0f);

        // a transposed weight view works too
        const Tensor transposed = buffer.transpose(0, 1);   // [3, 4]
        const Linear other(transposed);
        CHECK_EQ(other.in_features(), size_t{4});
        CHECK_EQ(other.out_features(), size_t{3});
        CHECK_EQ(other.forward(Tensor{Shape({1, 4})}).shape(), Shape({1, 3}));
    }

    // refusals
    {
        const Linear projection(weight());

        CHECK_THROWS_AS(projection.forward(Tensor{Shape({1, 5})}), std::invalid_argument);
        CHECK_THROWS_AS(projection.forward(Tensor{Shape({3})}), std::invalid_argument);
        try
        {
            (void)projection.forward(Tensor{Shape({2, 5})});
        }
        catch (const std::invalid_argument& error)
        {
            const std::string message = error.what();
            CHECK(message.find("(2, 5)") != std::string::npos);
            CHECK(message.find("in_features 3") != std::string::npos);
        }

        CHECK_THROWS_AS(Linear(Tensor{Shape({3})}), std::invalid_argument);
        CHECK_THROWS_AS(Linear(Tensor{Shape({2, 2, 2})}), std::invalid_argument);

        // a bias of the wrong length, or the wrong rank
        CHECK_THROWS_AS(Linear(weight(), Tensor{Shape({3})}), std::invalid_argument);
        CHECK_THROWS_AS(Linear(weight(), Tensor{Shape({2, 1})}), std::invalid_argument);
        try
        {
            (void)Linear(weight(), Tensor{Shape({5})});
        }
        catch (const std::invalid_argument& error)
        {
            CHECK(std::string(error.what()).find("[2]") != std::string::npos);
        }
    }

    // degenerate shapes
    {
        // a single output feature
        const Linear single_out(filled(Shape({1, 3}), {1, 1, 1}));
        CHECK_EQ(single_out.forward(filled(Shape({1, 3}), {1, 2, 3})).at({0, 0}), 6.0f);

        // a single input feature
        const Linear single_in(filled(Shape({2, 1}), {2, 3}));
        const Tensor y = single_in.forward(filled(Shape({1, 1}), {5}));
        CHECK_EQ(y.at({0, 0}), 10.0f);
        CHECK_EQ(y.at({0, 1}), 15.0f);

        // an empty sequence
        CHECK_EQ(Linear(weight()).forward(Tensor{Shape({0, 3})}).shape(), Shape({0, 2}));
    }

    return VEDA_TEST_SUMMARY("LinearTest");
}
