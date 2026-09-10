// E9.S1.T2 — FeedForward

#include "Activations.h"
#include "FeedForward.h"
#include "Linear.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::model::FeedForward;
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
} // namespace

int main()
{
    // the hand-computed example: D = 2, F = 3
    //   W_gate = [[1,0],[0,1],[1,1]]   x = [2,-2]  ->  gate_raw = [2,-2,0]
    //   silu([2,-2,0]) = [1.7616, -0.2384, 0]
    {
        const Tensor gate_w = filled(Shape({3, 2}), {1, 0, 0, 1, 1, 1});
        const Tensor down_w = filled(Shape({2, 3}), {1, 0, 1, 0, 1, 0});

        // up = 0: the gate says one thing, the other branch closes everything
        {
            const FeedForward ffn({Linear(gate_w), Linear(Tensor{Shape({3, 2})}), Linear(down_w)});
            const Tensor out = ffn.forward(filled(Shape({1, 2}), {2, -2}));
            CHECK_EQ(out.shape(), Shape({1, 2}));
            CHECK_EQ(out.at({0, 0}), 0.0f);
            CHECK_EQ(out.at({0, 1}), 0.0f);
        }

        // up = [1,1,1] for this input: two channels of the wide middle survive
        {
            const Tensor up_w = filled(Shape({3, 2}), {0.5f, 0, 0.5f, 0, 0.5f, 0});   // up = 0.5*x0
            const FeedForward ffn({Linear(gate_w), Linear(up_w), Linear(down_w)});
            const Tensor out = ffn.forward(filled(Shape({1, 2}), {2, -2}));

            // up = [1,1,1], gate = [1.7616, -0.2384, 0], gated = gate
            CHECK_NEAR(out.at({0, 0}), 1.7616, 1e-3);    // 1.7616*1 + (-0.2384)*0 + 0*1
            CHECK_NEAR(out.at({0, 1}), -0.2384, 1e-3);   // 1.7616*0 + (-0.2384)*1 + 0*0

            CHECK_EQ(ffn.hidden_size(), size_t{2});
            CHECK_EQ(ffn.inner_size(), size_t{3});
        }
    }

    // *** zeroed weights give exactly zero — the precondition for the block's identity test ***
    {
        const FeedForward ffn({Linear(Tensor{Shape({3, 2})}), Linear(Tensor{Shape({3, 2})}),
                               Linear(Tensor{Shape({2, 3})})});
        const Tensor out = ffn.forward(filled(Shape({1, 2, 2}), {1, 2, 3, 4}));

        CHECK_EQ(out.shape(), Shape({1, 2, 2}));
        for (size_t t = 0; t < 2; ++t)
        {
            for (size_t d = 0; d < 2; ++d)
            {
                CHECK_EQ(out.at({0, t, d}), 0.0f);
            }
        }
    }

    // *** the branch asymmetry: silu goes on the gate, and swapping the branches changes things ***
    {
        const Tensor gate_w = filled(Shape({2, 2}), {1, 0, 0, 1});
        const Tensor up_w = filled(Shape({2, 2}), {2, 0, 0, 2});
        const Tensor down_w = filled(Shape({2, 2}), {1, 0, 0, 1});

        const Tensor x = filled(Shape({1, 2}), {1, -1});

        const FeedForward ffn({Linear(gate_w), Linear(up_w), Linear(down_w)});
        const FeedForward swapped({Linear(up_w), Linear(gate_w), Linear(down_w)});

        const Tensor a = ffn.forward(x);
        const Tensor b = swapped.forward(x);

        // silu(x) * 2x  vs  silu(2x) * x — different functions
        CHECK_NEAR(a.at({0, 0}), veda::ops::silu(filled(Shape({1}), {1.0f})).at({0}) * 2.0f, 1e-4);
        CHECK_NEAR(b.at({0, 0}), veda::ops::silu(filled(Shape({1}), {2.0f})).at({0}) * 1.0f, 1e-4);
        CHECK(std::fabs(a.at({0, 0}) - b.at({0, 0})) > 1e-3f);
    }

    // shapes: positions are independent, so any leading shape passes through
    {
        const FeedForward ffn({Linear(Tensor{Shape({6, 4})}), Linear(Tensor{Shape({6, 4})}),
                               Linear(Tensor{Shape({4, 6})})});

        CHECK_EQ(ffn.forward(Tensor{Shape({8, 4})}).shape(), Shape({8, 4}));
        CHECK_EQ(ffn.forward(Tensor{Shape({1, 8, 4})}).shape(), Shape({1, 8, 4}));
        CHECK_EQ(ffn.forward(Tensor{Shape({2, 3, 8, 4})}).shape(), Shape({2, 3, 8, 4}));
        CHECK_EQ(ffn.forward(Tensor{Shape({0, 4})}).numel(), size_t{0});
        CHECK_THROWS_AS(ffn.forward(Tensor{Shape({1, 5})}), std::invalid_argument);
    }

    // the shape contract, checked at construction
    {
        // gate and up disagree on F
        CHECK_THROWS_AS(FeedForward({Linear(Tensor{Shape({3, 2})}), Linear(Tensor{Shape({4, 2})}),
                                     Linear(Tensor{Shape({2, 3})})}),
                        std::invalid_argument);

        // down_proj takes the wrong inner width
        CHECK_THROWS_AS(FeedForward({Linear(Tensor{Shape({3, 2})}), Linear(Tensor{Shape({3, 2})}),
                                     Linear(Tensor{Shape({2, 5})})}),
                        std::invalid_argument);

        // down_proj produces the wrong hidden width
        CHECK_THROWS_AS(FeedForward({Linear(Tensor{Shape({3, 2})}), Linear(Tensor{Shape({3, 2})}),
                                     Linear(Tensor{Shape({7, 3})})}),
                        std::invalid_argument);

        try
        {
            FeedForward bad({Linear(Tensor{Shape({3, 2})}), Linear(Tensor{Shape({4, 2})}),
                             Linear(Tensor{Shape({2, 3})})});
        }
        catch (const std::invalid_argument& error)
        {
            const std::string message = error.what();
            CHECK(message.find("3") != std::string::npos);
            CHECK(message.find("4") != std::string::npos);
        }
    }

    // AD2: weights are shared, not copied; the input is unmodified
    {
        const Tensor gate_w = filled(Shape({2, 2}), {1, 0, 0, 1});
        const FeedForward ffn({Linear(gate_w), Linear(gate_w), Linear(gate_w)});
        CHECK_EQ(gate_w.storage().use_count(), long{4});   // the caller plus three layers

        const Tensor x = filled(Shape({1, 2}), {3, 4});
        const Tensor out = ffn.forward(x);
        CHECK_EQ(x.at({0, 0}), 3.0f);
        CHECK(out.storage() != x.storage());
    }

    // the real proportions: 264M of Qwen3-0.6B's parameters are in these three matrices
    {
        const size_t D = 1024;
        const size_t F = 3072;
        const size_t per_layer = 3 * F * D;
        CHECK_EQ(per_layer, size_t{9'437'184});
        CHECK_EQ(per_layer * 28, size_t{264'241'152});

        // and the intermediate at T = 1024
        CHECK_EQ(1024 * F * sizeof(float), size_t{12'582'912});   // 12 MB
    }

    // degenerate widths
    {
        const FeedForward narrow({Linear(filled(Shape({1, 1}), {1})), Linear(filled(Shape({1, 1}), {1})),
                                  Linear(filled(Shape({1, 1}), {1}))});
        CHECK_NEAR(narrow.forward(filled(Shape({1, 1}), {2.0f})).at({0, 0}),
                   veda::ops::silu(filled(Shape({1}), {2.0f})).at({0}) * 2.0f, 1e-4);
    }

    return VEDA_TEST_SUMMARY("FeedForwardTest");
}
