// E16.S2.T3 — the trainable model: the same architecture, on the tape

#include "AdamW.h"
#include "Clip.h"
#include "GradientCheck.h"
#include "Loss.h"
#include "Model.h"
#include "ModelConfig.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"
#include "Variable.h"

#include <cmath>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

using veda::autograd::check_gradient;
using veda::autograd::cross_entropy;
using veda::autograd::Variable;
using veda::core::Shape;
using veda::core::Tensor;
using veda::optim::AdamW;
using veda::optim::AdamWConfig;
using veda::optim::clip_gradient_norm;
using veda::train::TinyTransformer;
using veda::train::tiny_config;

namespace
{
// Small enough that a full gradient check finishes, and still a complete transformer.
veda::io::ModelConfig test_config()
{
    veda::io::ModelConfig config = tiny_config();
    config.hidden_size = 8;
    config.intermediate_size = 16;
    config.num_layers = 2;
    config.num_heads = 2;
    config.num_kv_heads = 2;
    config.head_dim = 4;
    config.vocab_size = 11;
    config.max_position_embeddings = 16;
    return config;
}
} // namespace

int main()
{
    // --- the parameter layout is the contract with E4's loader ----------------------------------------
    {
        TinyTransformer model(tiny_config(), 1);
        const auto named = model.named_parameters();

        std::set<std::string> names;
        for (const auto& [name, tensor] : named)
        {
            names.insert(name);
        }
        CHECK_EQ(names.size(), named.size());   // no duplicates

        // exactly what Binding::bind asks for
        CHECK(names.count("model.embed_tokens.weight") == 1);
        CHECK(names.count("model.norm.weight") == 1);
        CHECK(names.count("model.layers.0.self_attn.q_proj.weight") == 1);
        CHECK(names.count("model.layers.0.self_attn.k_norm.weight") == 1);
        CHECK(names.count("model.layers.3.mlp.down_proj.weight") == 1);
        CHECK(names.count("model.layers.3.post_attention_layernorm.weight") == 1);
        CHECK(names.count("model.layers.4.mlp.up_proj.weight") == 0);   // four layers, not five
        // tied embeddings: the file must NOT contain lm_head.weight
        CHECK(names.count("lm_head.weight") == 0);

        // 1 embedding + 11 per layer * 4 + 1 final norm
        CHECK_EQ(named.size(), size_t{1 + 11 * 4 + 1});

        // and the shapes the loader will check them against
        const size_t D = 128, F = 512, H = 4, Dh = 32, V = 256;
        for (const auto& [name, tensor] : named)
        {
            if (name == "model.embed_tokens.weight")
            {
                CHECK_EQ(tensor.shape(), Shape({V, D}));
            }
            else if (name == "model.layers.2.self_attn.o_proj.weight")
            {
                CHECK_EQ(tensor.shape(), Shape({D, H * Dh}));
            }
            else if (name == "model.layers.2.mlp.gate_proj.weight")
            {
                CHECK_EQ(tensor.shape(), Shape({F, D}));
            }
            else if (name == "model.layers.2.self_attn.q_norm.weight")
            {
                CHECK_EQ(tensor.shape(), Shape({Dh}));
            }
        }

        // ~1.08 M parameters, as the arithmetic in the task document says
        size_t total = 0;
        for (const auto& [name, tensor] : named)
        {
            total += tensor.numel();
        }
        CHECK(total > 1'000'000);
        CHECK(total < 1'200'000);
    }

    // --- initialisation ------------------------------------------------------------------------------
    {
        TinyTransformer a(tiny_config(), 4242);
        TinyTransformer b(tiny_config(), 4242);
        TinyTransformer c(tiny_config(), 4243);

        const auto first = a.named_parameters();
        const auto same = b.named_parameters();
        const auto other = c.named_parameters();

        bool identical = true;
        bool differs = false;
        for (size_t i = 0; i < first.size(); ++i)
        {
            for (size_t e = 0; e < first[i].second.numel(); ++e)
            {
                identical = identical && first[i].second.data()[e] == same[i].second.data()[e];
                differs = differs || first[i].second.data()[e] != other[i].second.data()[e];
            }
        }
        CHECK(identical);   // same seed, same weights
        CHECK(differs);

        // *** norm weights start at exactly 1 — the identity gain; projections do not ***
        bool norms_are_one = true;
        bool projections_vary = false;
        for (const auto& [name, tensor] : first)
        {
            const bool is_norm = name.find("norm") != std::string::npos;
            for (size_t e = 0; e < tensor.numel(); ++e)
            {
                if (is_norm)
                {
                    norms_are_one = norms_are_one && tensor.data()[e] == 1.0f;
                }
                else if (tensor.data()[e] != 0.0f)
                {
                    projections_vary = true;
                }
            }
        }
        CHECK(norms_are_one);
        CHECK(projections_vary);

        // and the scale is 1/sqrt(fan_in), not something arbitrary
        for (const auto& [name, tensor] : first)
        {
            if (name == "model.layers.0.mlp.gate_proj.weight")
            {
                float largest = 0.0f;
                for (size_t e = 0; e < tensor.numel(); ++e)
                {
                    largest = std::fabs(tensor.data()[e]) > largest ? std::fabs(tensor.data()[e])
                                                                    : largest;
                }
                CHECK(largest <= 1.0f / std::sqrt(128.0f) + 1e-6f);
                CHECK(largest > 0.5f / std::sqrt(128.0f));
            }
        }
    }

    // --- the forward -----------------------------------------------------------------------------------
    {
        const auto config = test_config();
        TinyTransformer model(config, 7);

        const size_t B = 2, T = 5;
        std::vector<int64_t> ids(B * T);
        for (size_t i = 0; i < ids.size(); ++i)
        {
            ids[i] = static_cast<int64_t>(i % config.vocab_size);
        }

        Variable logits = model.forward(ids, B, T);
        CHECK_EQ(logits.value().shape(), Shape({B, T, config.vocab_size}));
        CHECK(logits.requires_grad());

        // *** an untrained model's loss is about log V — the first sanity check of any run ***
        std::vector<int64_t> targets(B * T, 3);
        const double loss = cross_entropy(logits, targets).value().at({});
        CHECK_NEAR(loss, std::log(static_cast<double>(config.vocab_size)), 0.5);

        // refusals
        CHECK_THROWS_AS(model.forward(ids, B, T + 1), std::invalid_argument);
        CHECK_THROWS_AS(model.forward(std::vector<int64_t>(100, 0), 1, 100),
                        std::invalid_argument);
    }

    // *** causality: a later token cannot change an earlier position's logits ***
    {
        // What a shape test cannot catch. If the mask were absent, or applied to the wrong axis,
        // every logit here would move.
        const auto config = test_config();
        TinyTransformer model(config, 11);

        const size_t T = 6;
        std::vector<int64_t> ids = {1, 2, 3, 4, 5, 6};
        std::vector<int64_t> changed = ids;
        changed[4] = 9;   // a change at position 4

        const Variable before = model.forward(ids, 1, T);
        const Variable after = model.forward(changed, 1, T);

        bool earlier_unchanged = true;
        for (size_t t = 0; t < 4; ++t)
        {
            for (size_t c = 0; c < config.vocab_size; ++c)
            {
                earlier_unchanged = earlier_unchanged &&
                                    before.value().at({0, t, c}) == after.value().at({0, t, c});
            }
        }
        CHECK(earlier_unchanged);

        // and the change did reach the positions it should have — otherwise the test proves nothing
        bool later_changed = false;
        for (size_t c = 0; c < config.vocab_size; ++c)
        {
            later_changed = later_changed ||
                            before.value().at({0, 4, c}) != after.value().at({0, 4, c});
        }
        CHECK(later_changed);
    }

    // *** finite differences through the whole assembled model ***
    {
        // Slow, and the single test that says the forward pass is differentiable end to end.
        const auto config = test_config();
        TinyTransformer model(config, 3);

        const std::vector<int64_t> ids = {1, 4, 2, 7};
        const std::vector<int64_t> targets = {4, 2, 7, 0};

        Variable loss = cross_entropy(model.forward(ids, 1, 4), targets);
        loss.backward();

        bool shapes_match = true;
        bool something_nonzero = false;
        for (const Variable& parameter : model.parameters())
        {
            shapes_match = shapes_match && parameter.grad().shape() == parameter.value().shape();
            for (size_t i = 0; i < parameter.grad().numel(); ++i)
            {
                something_nonzero = something_nonzero || parameter.grad().data()[i] != 0.0f;
                shapes_match = shapes_match && std::isfinite(parameter.grad().data()[i]);
            }
        }
        CHECK(shapes_match);
        CHECK(something_nonzero);

        // the real check: perturb one weight and compare against its analytic gradient
        Variable weight = model.parameters()[3];   // a k_proj
        const size_t index = 5;
        const float analytic = weight.grad().data()[index];
        const float original = weight.value().data()[index];
        const float h = 1e-2f;

        weight.node()->value.data()[index] = original + h;
        const double up = cross_entropy(model.forward(ids, 1, 4), targets).value().at({});
        weight.node()->value.data()[index] = original - h;
        const double down = cross_entropy(model.forward(ids, 1, 4), targets).value().at({});
        weight.node()->value.data()[index] = original;

        const double numeric = (up - down) / (2.0 * h);
        CHECK_NEAR(analytic, numeric, 2e-2 + 2e-2 * std::fabs(numeric));
    }

    // --- the milestone in miniature: one fixed batch, memorised -------------------------------------
    {
        const auto config = test_config();
        TinyTransformer model(config, 5);

        const size_t B = 2, T = 6;
        std::vector<int64_t> ids(B * T);
        std::vector<int64_t> targets(B * T);
        for (size_t i = 0; i < ids.size(); ++i)
        {
            ids[i] = static_cast<int64_t>((i * 3 + 1) % config.vocab_size);
            targets[i] = static_cast<int64_t>((i * 5 + 2) % config.vocab_size);
        }

        AdamW optimiser(model.parameters(), AdamWConfig{0.02f, 0.9f, 0.999f, 1e-8f, 0.0f});

        const double before = cross_entropy(model.forward(ids, B, T), targets).value().at({});
        for (int step = 0; step < 120; ++step)
        {
            optimiser.zero_grad();
            cross_entropy(model.forward(ids, B, T), targets).backward();
            clip_gradient_norm(model.parameters(), 1.0f);
            optimiser.step();
        }
        const double after = cross_entropy(model.forward(ids, B, T), targets).value().at({});

        CHECK_NEAR(before, std::log(static_cast<double>(config.vocab_size)), 0.6);
        CHECK(after < before);
        CHECK(after < 0.1);   // *** the model can memorise — the precondition for learning ***
    }

    return VEDA_TEST_SUMMARY("TinyModelTest");
}
