// E10.S1.T2 — weight binding

#include "Binding.h"
#include "Json.h"
#include "ModelConfig.h"
#include "Safetensors.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <stdexcept>
#include <string>
#include <vector>

using veda::binding::bind;
using veda::core::Shape;
using veda::core::Tensor;
using veda::io::ModelConfig;
using veda::io::WeightFile;

namespace
{
// A tiny Qwen3-shaped model: D = 8, F = 12, H = 4, Hkv = 2, Dh = 2, V = 16, 2 layers.
ModelConfig small_config(bool tied = true)
{
    ModelConfig config;
    config.hidden_size = 8;
    config.intermediate_size = 12;
    config.num_layers = 2;
    config.num_heads = 4;
    config.num_kv_heads = 2;
    config.head_dim = 2;
    config.vocab_size = 16;
    config.max_position_embeddings = 64;
    config.rms_norm_eps = 1e-6f;
    config.rope_theta = 1000000.0f;
    config.tie_word_embeddings = tied;
    return config;
}

Tensor of(const Shape& shape)
{
    Tensor t{shape};
    for (size_t i = 0; i < t.numel(); ++i)
    {
        t.data()[i] = 0.01f * static_cast<float>(i + 1);
    }
    return t;
}

// The file a real export would produce for that config. Built in memory: a binding test should not
// depend on the safetensors reader.
WeightFile weight_file(const ModelConfig& config, const std::string& skip = "",
                       const std::string& extra = "")
{
    const size_t D = config.hidden_size;
    const size_t F = config.intermediate_size;
    const size_t Dh = config.head_dim;
    const size_t H = config.num_heads;
    const size_t Hkv = config.num_kv_heads;

    WeightFile weights;
    auto add = [&](const std::string& name, const Shape& shape) {
        if (name != skip)
        {
            weights.add(name, of(shape), shape.size() * 4);
        }
    };

    add("model.embed_tokens.weight", Shape({config.vocab_size, D}));
    for (size_t layer = 0; layer < config.num_layers; ++layer)
    {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        add(prefix + "input_layernorm.weight", Shape({D}));
        add(prefix + "self_attn.q_proj.weight", Shape({H * Dh, D}));
        add(prefix + "self_attn.k_proj.weight", Shape({Hkv * Dh, D}));
        add(prefix + "self_attn.v_proj.weight", Shape({Hkv * Dh, D}));
        add(prefix + "self_attn.o_proj.weight", Shape({D, H * Dh}));
        add(prefix + "self_attn.q_norm.weight", Shape({Dh}));
        add(prefix + "self_attn.k_norm.weight", Shape({Dh}));
        add(prefix + "post_attention_layernorm.weight", Shape({D}));
        add(prefix + "mlp.gate_proj.weight", Shape({F, D}));
        add(prefix + "mlp.up_proj.weight", Shape({F, D}));
        add(prefix + "mlp.down_proj.weight", Shape({D, F}));
    }
    add("model.norm.weight", Shape({D}));
    if (!config.tie_word_embeddings)
    {
        add("lm_head.weight", Shape({config.vocab_size, D}));
    }
    if (!extra.empty())
    {
        add(extra, Shape({4}));
    }
    return weights;
}

std::string message_of(const WeightFile& weights, const ModelConfig& config)
{
    try
    {
        (void)bind(weights, config);
    }
    catch (const std::runtime_error& error)
    {
        return error.what();
    }
    return "";
}
} // namespace

int main()
{
    // a complete file binds
    {
        const ModelConfig config = small_config();
        const WeightFile weights = weight_file(config);

        // 1 embedding + 2 layers * 11 + 1 final norm
        CHECK_EQ(weights.tensor_count(), size_t{24});

        const veda::model::Transformer model = bind(weights, config);
        CHECK_EQ(model.blocks().size(), size_t{2});
        CHECK_EQ(model.embedding().vocab_size(), size_t{16});
        CHECK_EQ(model.embedding().hidden_size(), size_t{8});
        CHECK_EQ(model.blocks()[0].hidden_size(), size_t{8});
        CHECK_EQ(model.final_norm().hidden_size(), size_t{8});
        CHECK_NEAR(model.final_norm().eps(), 1e-6, 1e-12);

        // and it runs
        const Tensor x{Shape({1, 3, 8})};
        CHECK_EQ(model.blocks()[0].forward(x).shape(), Shape({1, 3, 8}));
        CHECK_EQ(model.final_norm().forward(x).shape(), Shape({1, 3, 8}));
        CHECK_EQ(model.embedding().forward({0, 5, 15}, Shape({1, 3})).shape(), Shape({1, 3, 8}));
    }

    // *** nothing is copied: every layer holds a view into the file's storage (AD2) ***
    {
        const ModelConfig config = small_config();
        const WeightFile weights = weight_file(config);
        const veda::model::Transformer model = bind(weights, config);

        CHECK(model.embedding().table().storage() ==
              weights["model.embed_tokens.weight"].storage());
        CHECK(model.final_norm().gamma().storage() == weights["model.norm.weight"].storage());
        CHECK(model.embedding().table().data() == weights["model.embed_tokens.weight"].data());
    }

    // *** direction one: a missing tensor is named ***
    {
        const ModelConfig config = small_config();
        const std::string missing = "model.layers.1.mlp.up_proj.weight";
        const WeightFile weights = weight_file(config, missing);

        CHECK_THROWS_AS(bind(weights, config), std::runtime_error);
        const std::string message = message_of(weights, config);
        CHECK(message.find("no tensor named") != std::string::npos);
        CHECK(message.find(missing) != std::string::npos);

        // and the Qwen3-specific ones are required, not optional: a binding written from a Llama
        // tutorial would not ask for q_norm at all
        const WeightFile without_qnorm =
            weight_file(config, "model.layers.0.self_attn.q_norm.weight");
        CHECK(message_of(without_qnorm, config).find("q_norm") != std::string::npos);
    }

    // *** direction two: an unconsumed tensor is named — the half people skip ***
    {
        const ModelConfig config = small_config();
        const WeightFile weights =
            weight_file(config, "", "model.layers.0.self_attn.rotary_emb.inv_freq");

        CHECK_THROWS_AS(bind(weights, config), std::runtime_error);
        const std::string message = message_of(weights, config);
        CHECK(message.find("were not used") != std::string::npos);
        CHECK(message.find("rotary_emb.inv_freq") != std::string::npos);
        CHECK(message.find("1 tensor") != std::string::npos);
    }

    // a shape that disagrees with the config is caught at bind time, not in layer 7
    {
        const ModelConfig config = small_config();
        WeightFile weights = weight_file(config, "model.layers.0.self_attn.q_proj.weight");
        weights.add("model.layers.0.self_attn.q_proj.weight", of(Shape({4, 8})), 128);   // should be [8,8]

        CHECK_THROWS_AS(bind(weights, config), std::runtime_error);
        const std::string message = message_of(weights, config);
        CHECK(message.find("q_proj") != std::string::npos);
        CHECK(message.find("(4, 8)") != std::string::npos);
        CHECK(message.find("(8, 8)") != std::string::npos);
    }

    // tied embeddings: absent lm_head.weight is correct, present is an error
    {
        const ModelConfig tied = small_config(true);
        CHECK_EQ(bind(weight_file(tied), tied).blocks().size(), size_t{2});

        // a tied config whose file nevertheless carries lm_head.weight has an unused tensor
        const WeightFile with_head = weight_file(tied, "", "lm_head.weight");
        CHECK(message_of(with_head, tied).find("were not used") != std::string::npos);
    }

    // untied embeddings: lm_head.weight must be there, and is consumed
    {
        const ModelConfig untied = small_config(false);
        const WeightFile complete = weight_file(untied);
        CHECK_EQ(complete.tensor_count(), size_t{25});
        CHECK_EQ(bind(complete, untied).blocks().size(), size_t{2});

        const WeightFile without = weight_file(untied, "lm_head.weight");
        CHECK(message_of(without, untied).find("lm_head.weight") != std::string::npos);
    }

    // GQA: the q and k widths differ, and the binding checks each against the config
    {
        const ModelConfig config = small_config();
        const WeightFile weights = weight_file(config);

        CHECK_EQ(weights["model.layers.0.self_attn.q_proj.weight"].shape(), Shape({8, 8}));
        CHECK_EQ(weights["model.layers.0.self_attn.k_proj.weight"].shape(), Shape({4, 8}));

        const veda::model::Transformer model = bind(weights, config);
        CHECK_EQ(model.blocks()[0].forward(Tensor{Shape({1, 2, 8})}).shape(), Shape({1, 2, 8}));
    }

    // the real Qwen3-0.6B tensor count, as arithmetic
    {
        const size_t layers = 28;
        const size_t per_layer = 11;
        CHECK_EQ(1 + layers * per_layer + 1, size_t{310});
    }

    return VEDA_TEST_SUMMARY("BindingTest");
}
