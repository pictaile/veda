// E10.S2.T3 — the stack

#include "Binding.h"
#include "Compare.h"
#include "Embedding.h"
#include "FeedForward.h"
#include "Linear.h"
#include "ModelConfig.h"
#include "RMSNorm.h"
#include "Safetensors.h"
#include "ScaledDotProductAttention.h"
#include "Shape.h"
#include "TensorFile.h"
#include "TestSupport.h"
#include "Tensor.h"
#include "Transformer.h"
#include "TransformerBlock.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::model::FeedForward;
using veda::model::ScaledDotProductAttention;
using veda::model::Transformer;
using veda::model::TransformerBlock;
using veda::nn::Embedding;
using veda::nn::Linear;
using veda::nn::RMSNorm;

namespace
{
const float eps = 1e-6f;

Tensor ones(size_t n)
{
    Tensor t{Shape({n})};
    for (size_t i = 0; i < n; ++i)
    {
        t.data()[i] = 1.0f;
    }
    return t;
}

Tensor identity(size_t n)
{
    Tensor t{Shape({n, n})};
    for (size_t i = 0; i < n; ++i)
    {
        t.data()[i * n + i] = 1.0f;
    }
    return t;
}

Tensor counting(const Shape& shape, float scale = 0.1f)
{
    Tensor t{shape};
    for (size_t i = 0; i < shape.size(); ++i)
    {
        t.data()[i] = scale * static_cast<float>(i + 1);
    }
    return t;
}

// A block whose projections are all zero: by E9's identity property it passes its input through.
TransformerBlock zeroed_block(size_t D, size_t F)
{
    return TransformerBlock({RMSNorm(ones(D), eps),
                             ScaledDotProductAttention(
                                 {Linear(Tensor{Shape({D, D})}), Linear(Tensor{Shape({D, D})}),
                                  Linear(Tensor{Shape({D, D})}), Linear(Tensor{Shape({D, D})})},
                                 {1, 1, D, std::nullopt}),
                             RMSNorm(ones(D), eps),
                             FeedForward({Linear(Tensor{Shape({F, D})}), Linear(Tensor{Shape({F, D})}),
                                          Linear(Tensor{Shape({D, F})})})});
}

TransformerBlock live_block(size_t D, size_t F, std::optional<float> rope = std::nullopt)
{
    return TransformerBlock({RMSNorm(ones(D), eps),
                             ScaledDotProductAttention({Linear(identity(D)), Linear(identity(D)),
                                                        Linear(identity(D)), Linear(identity(D))},
                                                       {1, 1, D, rope}),
                             RMSNorm(ones(D), eps),
                             FeedForward({Linear(counting(Shape({F, D}))),
                                          Linear(counting(Shape({F, D}))),
                                          Linear(counting(Shape({D, F})))})});
}
} // namespace

int main()
{
    const size_t V = 6;
    const size_t D = 4;
    const size_t F = 6;

    // the stack runs end to end
    {
        std::vector<TransformerBlock> blocks;
        blocks.push_back(live_block(D, F));
        blocks.push_back(live_block(D, F));

        const Transformer transformer(
            {Embedding(counting(Shape({V, D}))), std::move(blocks), RMSNorm(ones(D), eps)});

        CHECK_EQ(transformer.blocks().size(), size_t{2});
        CHECK_EQ(transformer.hidden_size(), D);

        const Tensor hidden = transformer.forward({0, 3, 5}, Shape({1, 3}));
        CHECK_EQ(hidden.shape(), Shape({1, 3, D}));
        CHECK(hidden.is_contiguous());
    }

    // *** zeroed blocks: the stack is final_norm(embedding(ids)), exactly ***
    // Every block is the identity (E9.S2.T3), so anything else means the loop is wrong.
    {
        std::vector<TransformerBlock> blocks;
        for (int i = 0; i < 3; ++i)
        {
            blocks.push_back(zeroed_block(D, F));
        }

        const Embedding embedding(counting(Shape({V, D})));
        const RMSNorm final_norm(ones(D), eps);
        const Transformer transformer({embedding, std::move(blocks), final_norm});

        const std::vector<int64_t> ids = {1, 4};
        const Tensor through_stack = transformer.forward(ids, Shape({1, 2}));
        const Tensor direct = final_norm.forward(embedding.forward(ids, Shape({1, 2})));

        CHECK_EQ(through_stack.shape(), direct.shape());
        for (size_t t = 0; t < 2; ++t)
        {
            for (size_t d = 0; d < D; ++d)
            {
                CHECK_EQ(through_stack.at({0, t, d}), direct.at({0, t, d}));
            }
        }
    }

    // *** the trace: one embedding, one entry per block, one final hidden ***
    {
        std::vector<TransformerBlock> blocks;
        for (int i = 0; i < 3; ++i)
        {
            blocks.push_back(live_block(D, F));
        }

        const RMSNorm final_norm(ones(D), eps);
        const Transformer transformer(
            {Embedding(counting(Shape({V, D}))), std::move(blocks), final_norm});

        const std::vector<int64_t> ids = {0, 2, 5};
        const auto trace = transformer.forward_capturing(ids, Shape({1, 3}));

        CHECK_EQ(trace.layer_outputs.size(), size_t{3});
        CHECK_EQ(trace.embeddings.shape(), Shape({1, 3, D}));
        CHECK_EQ(trace.final_hidden.shape(), Shape({1, 3, D}));
        for (const Tensor& layer : trace.layer_outputs)
        {
            CHECK_EQ(layer.shape(), Shape({1, 3, D}));
        }

        // the last block's output feeds the final norm, not the other way round
        const Tensor expected_final = final_norm.forward(trace.layer_outputs.back());
        for (size_t t = 0; t < 3; ++t)
        {
            for (size_t d = 0; d < D; ++d)
            {
                CHECK_NEAR(trace.final_hidden.at({0, t, d}), expected_final.at({0, t, d}), 1e-6);
            }
        }

        // and the trace agrees with the plain forward
        const Tensor plain = transformer.forward(ids, Shape({1, 3}));
        for (size_t t = 0; t < 3; ++t)
        {
            for (size_t d = 0; d < D; ++d)
            {
                CHECK_EQ(plain.at({0, t, d}), trace.final_hidden.at({0, t, d}));
            }
        }

        // the residual stream accumulates: each layer's output differs from the last
        bool layers_change = false;
        for (size_t d = 0; d < D; ++d)
        {
            if (trace.layer_outputs[0].at({0, 2, d}) != trace.layer_outputs[2].at({0, 2, d}))
            {
                layers_change = true;
            }
        }
        CHECK(layers_change);
    }

    // the position reaches every block
    {
        std::vector<TransformerBlock> rotated_blocks;
        std::vector<TransformerBlock> plain_blocks;
        for (int i = 0; i < 2; ++i)
        {
            rotated_blocks.push_back(live_block(D, F, 10000.0f));
            plain_blocks.push_back(live_block(D, F));
        }

        const Transformer rotated(
            {Embedding(counting(Shape({V, D}))), std::move(rotated_blocks), RMSNorm(ones(D), eps)});
        const Transformer plain(
            {Embedding(counting(Shape({V, D}))), std::move(plain_blocks), RMSNorm(ones(D), eps)});

        const std::vector<int64_t> ids = {0, 1, 2};
        bool differs = false;
        for (size_t d = 0; d < D; ++d)
        {
            if (std::fabs(rotated.forward(ids, Shape({1, 3})).at({0, 2, d}) -
                          plain.forward(ids, Shape({1, 3})).at({0, 2, d})) > 1e-5f)
            {
                differs = true;
            }
        }
        CHECK(differs);
    }

    // shapes and degenerate cases
    {
        std::vector<TransformerBlock> blocks;
        blocks.push_back(live_block(D, F));
        const Transformer transformer(
            {Embedding(counting(Shape({V, D}))), std::move(blocks), RMSNorm(ones(D), eps)});

        CHECK_EQ(transformer.forward({0}, Shape({1, 1})).shape(), Shape({1, 1, D}));
        CHECK_EQ(transformer.forward({0, 1, 2, 3}, Shape({2, 2})).shape(), Shape({2, 2, D}));
        CHECK_EQ(transformer.forward({}, Shape({1, 0})).numel(), size_t{0});
        CHECK_THROWS_AS(transformer.forward({V}, Shape({1, 1})), std::out_of_range);

        // a zero-layer stack is legal
        const Transformer bare(
            {Embedding(counting(Shape({V, D}))), {}, RMSNorm(ones(D), eps)});
        CHECK_EQ(bare.forward({2}, Shape({1, 1})).shape(), Shape({1, 1, D}));
        CHECK_EQ(bare.forward_capturing({2}, Shape({1, 1})).layer_outputs.size(), size_t{0});
    }

    // binding produces one, and it runs
    {
        veda::io::ModelConfig config;
        config.hidden_size = 8;
        config.intermediate_size = 12;
        config.num_layers = 2;
        config.num_heads = 4;
        config.num_kv_heads = 2;
        config.head_dim = 2;
        config.vocab_size = 16;
        config.rms_norm_eps = 1e-6f;
        config.rope_theta = 1000000.0f;
        config.tie_word_embeddings = true;

        veda::io::WeightFile weights;
        auto add = [&](const std::string& name, const Shape& shape) {
            weights.add(name, counting(shape, 0.01f), shape.size() * 4);
        };
        add("model.embed_tokens.weight", Shape({16, 8}));
        for (size_t layer = 0; layer < 2; ++layer)
        {
            const std::string prefix = "model.layers." + std::to_string(layer) + ".";
            add(prefix + "input_layernorm.weight", Shape({8}));
            add(prefix + "self_attn.q_proj.weight", Shape({8, 8}));
            add(prefix + "self_attn.k_proj.weight", Shape({4, 8}));
            add(prefix + "self_attn.v_proj.weight", Shape({4, 8}));
            add(prefix + "self_attn.o_proj.weight", Shape({8, 8}));
            add(prefix + "self_attn.q_norm.weight", Shape({2}));
            add(prefix + "self_attn.k_norm.weight", Shape({2}));
            add(prefix + "post_attention_layernorm.weight", Shape({8}));
            add(prefix + "mlp.gate_proj.weight", Shape({12, 8}));
            add(prefix + "mlp.up_proj.weight", Shape({12, 8}));
            add(prefix + "mlp.down_proj.weight", Shape({8, 12}));
        }
        add("model.norm.weight", Shape({8}));

        const Transformer transformer = veda::binding::bind(weights, config);
        CHECK_EQ(transformer.blocks().size(), size_t{2});

        const Tensor hidden = transformer.forward({1, 2, 3, 4}, Shape({1, 4}));
        CHECK_EQ(hidden.shape(), Shape({1, 4, 8}));

        // every number is finite: a GQA + QK-Norm + RoPE stack that produced NaN would say so here
        bool finite = true;
        for (size_t i = 0; i < hidden.numel(); ++i)
        {
            if (!std::isfinite(hidden.data()[i]))
            {
                finite = false;
            }
        }
        CHECK(finite);

        // and the trace bisects it
        const auto trace = transformer.forward_capturing({1, 2, 3, 4}, Shape({1, 4}));
        CHECK_EQ(trace.layer_outputs.size(), size_t{2});
    }

    // --- the per-layer reference comparison, which is what Trace exists for --------------------
    {
        const std::filesystem::path directory = "reference";
        if (std::filesystem::exists(directory / "layer_00_output.bin"))
        {
            std::cout << "  reference found — per-layer comparison would run here (E10.S3)\n";
            const Tensor first = veda::testing::read_tensor((directory / "layer_00_output.bin").string());
            CHECK_EQ(first.rank(), size_t{3});
        }
        else
        {
            std::cout << "  SKIPPED: no reference/ — run tools/dump_reference.py against the real\n"
                      << "  model to bisect the stack layer by layer.\n";
        }
    }

    return VEDA_TEST_SUMMARY("TransformerTest");
}
