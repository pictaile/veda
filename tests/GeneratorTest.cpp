// E11.S3.T5 — the generation loop

#include "Binding.h"
#include "Generator.h"
#include "LMHead.h"
#include "ModelConfig.h"
#include "Safetensors.h"
#include "Sampling.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"
#include "Transformer.h"

#include <stdexcept>
#include <string>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::generate::GreedySampler;
using veda::generate::Generator;
using veda::generate::Options;
using veda::generate::Result;
using veda::generate::TemperatureSampler;
using veda::model::LMHead;
using veda::model::Transformer;

namespace
{
const size_t V = 12;
const size_t D = 8;

veda::io::ModelConfig small_config()
{
    veda::io::ModelConfig config;
    config.hidden_size = D;
    config.intermediate_size = 12;
    config.num_layers = 2;
    config.num_heads = 4;
    config.num_kv_heads = 2;
    config.head_dim = 2;
    config.vocab_size = V;
    config.rms_norm_eps = 1e-6f;
    config.rope_theta = 1000000.0f;
    config.tie_word_embeddings = true;
    return config;
}

Transformer build_model()
{
    const veda::io::ModelConfig config = small_config();
    veda::io::WeightFile weights;

    auto add = [&](const std::string& name, const Shape& shape, float scale) {
        Tensor t{shape};
        for (size_t i = 0; i < t.numel(); ++i)
        {
            t.data()[i] = scale * (0.01f * static_cast<float>((i * 7) % 23) - 0.1f);
        }
        weights.add(name, t, shape.size() * 4);
    };

    add("model.embed_tokens.weight", Shape({V, D}), 1.0f);
    for (size_t layer = 0; layer < 2; ++layer)
    {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        add(prefix + "input_layernorm.weight", Shape({D}), 1.0f);
        add(prefix + "self_attn.q_proj.weight", Shape({8, D}), 1.0f);
        add(prefix + "self_attn.k_proj.weight", Shape({4, D}), 1.0f);
        add(prefix + "self_attn.v_proj.weight", Shape({4, D}), 1.0f);
        add(prefix + "self_attn.o_proj.weight", Shape({D, 8}), 1.0f);
        add(prefix + "self_attn.q_norm.weight", Shape({2}), 1.0f);
        add(prefix + "self_attn.k_norm.weight", Shape({2}), 1.0f);
        add(prefix + "post_attention_layernorm.weight", Shape({D}), 1.0f);
        add(prefix + "mlp.gate_proj.weight", Shape({12, D}), 1.0f);
        add(prefix + "mlp.up_proj.weight", Shape({12, D}), 1.0f);
        add(prefix + "mlp.down_proj.weight", Shape({D, 12}), 1.0f);
    }
    add("model.norm.weight", Shape({D}), 1.0f);

    return veda::binding::bind(weights, config);
}

// The loop's behaviour should not depend on the model's internals, so the tests that check the
// loop fix the sampler instead: a rigged head would depend on the sign of the hidden state, which
// is exactly the kind of accidental coupling a test should not have.
class ConstantSampler final : public veda::generate::Sampler
{
public:
    explicit ConstantSampler(int32_t id) : id_(id) {}
    int32_t sample(const Tensor&) override { return id_; }

private:
    int32_t id_;
};

// A plain head over the tied embedding table, for the tests that want real logits.
LMHead tied_head(const Transformer& transformer)
{
    return LMHead(transformer.embedding().table());
}
} // namespace

int main()
{
    const Transformer transformer = build_model();

    // *** a fixed sampler makes the loop exactly predictable ***
    {
        const LMHead head = tied_head(transformer);
        ConstantSampler sampler(7);
        const Generator generator(transformer, head, sampler);

        const std::vector<int32_t> prompt = {1, 2, 3};
        const Result result = generator.generate(prompt, {5, {}});

        CHECK_EQ(result.generated.size(), size_t{5});
        CHECK_EQ(result.generated, std::vector<int32_t>({7, 7, 7, 7, 7}));
        CHECK(!result.stopped_on_token);

        // *** one forward pass per generated token — the number E12 will be measured against ***
        CHECK_EQ(result.forward_passes, result.generated.size());
    }

    // *** the stop token ends generation, and is not in the output ***
    {
        const LMHead head = tied_head(transformer);
        ConstantSampler sampler(9);
        const Generator generator(transformer, head, sampler);

        const Result result = generator.generate({1, 2}, {20, {9}});

        CHECK(result.generated.empty());          // the first token was the stop token
        CHECK(result.stopped_on_token);
        CHECK_EQ(result.forward_passes, size_t{1});

        // and a stop token that never comes leaves the limit in charge
        const Result other = generator.generate({1, 2}, {4, {11}});
        CHECK_EQ(other.generated.size(), size_t{4});
        CHECK(!other.stopped_on_token);
        CHECK_EQ(other.forward_passes, size_t{4});
    }

    // the two endings are distinguishable — a caller must be able to tell truncation from finishing
    {
        const LMHead head = tied_head(transformer);
        ConstantSampler sampler(3);
        const Generator generator(transformer, head, sampler);

        const Result truncated = generator.generate({5}, {2, {}});
        const Result finished = generator.generate({5}, {10, {3}});

        CHECK(!truncated.stopped_on_token);
        CHECK_EQ(truncated.generated.size(), size_t{2});
        CHECK(finished.stopped_on_token);
        CHECK(finished.generated.empty());
    }

    // the callback sees every emitted token, in order, and never the stop token
    {
        const LMHead head = tied_head(transformer);
        ConstantSampler sampler(4);
        const Generator generator(transformer, head, sampler);

        std::vector<int32_t> streamed;
        const Result result =
            generator.generate({1}, {3, {}}, [&](int32_t id) { streamed.push_back(id); });

        CHECK_EQ(streamed, result.generated);
        CHECK_EQ(streamed.size(), size_t{3});

        std::vector<int32_t> nothing;
        (void)generator.generate({1}, {5, {4}}, [&](int32_t id) { nothing.push_back(id); });
        CHECK(nothing.empty());   // the stop token was never handed to the callback
    }

    // *** determinism: the same seed and prompt reproduce the output ***
    {
        const LMHead head = tied_head(transformer);   // tied, so the logits are interesting

        TemperatureSampler first(0.9f, 4242);
        TemperatureSampler second(0.9f, 4242);
        TemperatureSampler different(0.9f, 1);

        const Generator a(transformer, head, first);
        const Generator b(transformer, head, second);
        const Generator c(transformer, head, different);

        const std::vector<int32_t> prompt = {2, 5, 1};
        const Result run_a = a.generate(prompt, {12, {}});
        const Result run_b = b.generate(prompt, {12, {}});
        const Result run_c = c.generate(prompt, {12, {}});

        CHECK_EQ(run_a.generated, run_b.generated);
        CHECK(run_a.generated != run_c.generated);
        CHECK_EQ(run_a.generated.size(), size_t{12});

        // every id is in range, so the model could be fed its own output
        for (const int32_t id : run_a.generated)
        {
            CHECK(id >= 0 && static_cast<size_t>(id) < V);
        }
    }

    // the prompt is not modified, and generation conditions on it
    {
        const LMHead head = tied_head(transformer);
        GreedySampler sampler;
        const Generator generator(transformer, head, sampler);

        const std::vector<int32_t> prompt = {3, 1, 4};
        const Result result = generator.generate(prompt, {3, {}});

        CHECK_EQ(prompt, std::vector<int32_t>({3, 1, 4}));

        // a different prompt gives a different continuation — the loop really conditions
        const Result other = generator.generate({9, 9, 9}, {3, {}});
        CHECK(result.generated != other.generated || result.generated.empty());
    }

    // edge cases
    {
        const LMHead head = tied_head(transformer);
        ConstantSampler sampler(2);
        const Generator generator(transformer, head, sampler);

        const Result nothing = generator.generate({1}, {0, {}});
        CHECK(nothing.generated.empty());
        CHECK_EQ(nothing.forward_passes, size_t{0});   // nothing ran at all
        CHECK(!nothing.stopped_on_token);

        CHECK_THROWS_AS(generator.generate({}, {5, {}}), std::invalid_argument);

        // a single-token prompt is legal
        CHECK_EQ(generator.generate({0}, {2, {}}).generated.size(), size_t{2});

        // several stop tokens
        const Result stopped = generator.generate({1}, {10, {5, 2, 8}});
        CHECK(stopped.stopped_on_token);
    }

    // what a run costs without a cache — the arithmetic behind risk R3
    {
        const size_t prompt_length = 25;
        const size_t new_tokens = 50;

        size_t position_passes = 0;
        for (size_t step = 0; step < new_tokens; ++step)
        {
            position_passes += prompt_length + step + 1;
        }

        CHECK_EQ(position_passes, size_t{2525});                    // every prefix, every step
        CHECK_EQ(prompt_length + new_tokens, size_t{75});           // what a cache would cost (E12)
        CHECK(position_passes / (prompt_length + new_tokens) > 30);
    }

    return VEDA_TEST_SUMMARY("GeneratorTest");
}
