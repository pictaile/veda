// E12.S1.T1 (the waste, measured) and T2 (the cache)

#include "Binding.h"
#include "Generator.h"
#include "KVCache.h"
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
using veda::generate::Result;
using veda::generate::TemperatureSampler;
using veda::model::KVCache;
using veda::model::LMHead;
using veda::model::Transformer;

namespace
{
const size_t V = 12;
const size_t D = 8;
const size_t LAYERS = 2;
const size_t HKV = 2;
const size_t DH = 2;

veda::io::ModelConfig small_config()
{
    veda::io::ModelConfig config;
    config.hidden_size = D;
    config.intermediate_size = 12;
    config.num_layers = LAYERS;
    config.num_heads = 4;
    config.num_kv_heads = HKV;
    config.head_dim = DH;
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

    auto add = [&](const std::string& name, const Shape& shape) {
        Tensor t{shape};
        for (size_t i = 0; i < t.numel(); ++i)
        {
            t.data()[i] = 0.01f * static_cast<float>((i * 7) % 23) - 0.1f;
        }
        weights.add(name, t, shape.size() * 4);
    };

    add("model.embed_tokens.weight", Shape({V, D}));
    for (size_t layer = 0; layer < LAYERS; ++layer)
    {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        add(prefix + "input_layernorm.weight", Shape({D}));
        add(prefix + "self_attn.q_proj.weight", Shape({8, D}));
        add(prefix + "self_attn.k_proj.weight", Shape({HKV * DH, D}));
        add(prefix + "self_attn.v_proj.weight", Shape({HKV * DH, D}));
        add(prefix + "self_attn.o_proj.weight", Shape({D, 8}));
        add(prefix + "self_attn.q_norm.weight", Shape({DH}));
        add(prefix + "self_attn.k_norm.weight", Shape({DH}));
        add(prefix + "post_attention_layernorm.weight", Shape({D}));
        add(prefix + "mlp.gate_proj.weight", Shape({12, D}));
        add(prefix + "mlp.up_proj.weight", Shape({12, D}));
        add(prefix + "mlp.down_proj.weight", Shape({D, 12}));
    }
    add("model.norm.weight", Shape({D}));

    return veda::binding::bind(weights, config);
}
} // namespace

int main()
{
    const Transformer transformer = build_model();
    const LMHead head(transformer.embedding().table());

    // --- T1: the waste, counted ------------------------------------------------------------------
    {
        const size_t prompt_length = 25;
        const size_t new_tokens = 50;

        size_t without = 0;
        for (size_t step = 0; step < new_tokens; ++step)
        {
            without += prompt_length + step + 1;
        }
        const size_t with_cache = prompt_length + new_tokens;

        CHECK_EQ(without, size_t{2525});
        CHECK_EQ(with_cache, size_t{75});
        CHECK_EQ(without / with_cache, size_t{33});

        // the waste grows quadratically while the useful work grows linearly
        auto passes = [](size_t T, size_t N) {
            size_t total = 0;
            for (size_t step = 0; step < N; ++step)
            {
                total += T + step + 1;
            }
            return total;
        };
        CHECK_EQ(passes(25, 10), size_t{305});      // 10*25 + 10*11/2
        CHECK_EQ(passes(25, 200), size_t{25100});   // 200*25 + 200*201/2
        CHECK_EQ(passes(25, 10) / (25 + 10), size_t{8});
        CHECK_EQ(passes(25, 200) / (25 + 200), size_t{111});

        // the cache's memory, at Qwen3-0.6B's numbers and a 32k context
        CHECK_EQ(2 * 8 * 32768 * 128 * sizeof(float) * 28, size_t{7'516'192'768});
    }

    // --- T2: the cache itself ---------------------------------------------------------------------
    {
        KVCache cache(LAYERS, 1, HKV, 16, DH);
        CHECK_EQ(cache.used(), size_t{0});
        CHECK_EQ(cache.layers(), LAYERS);
        CHECK_EQ(cache.max_positions(), size_t{16});
        CHECK_EQ(cache.keys(0).shape(), Shape({1, HKV, 0, DH}));

        Tensor k{Shape({1, HKV, 3, DH})};
        Tensor v{Shape({1, HKV, 3, DH})};
        for (size_t i = 0; i < k.numel(); ++i)
        {
            k.data()[i] = static_cast<float>(i);
            v.data()[i] = -static_cast<float>(i);
        }

        cache.append(0, k, v);
        cache.append(1, k, v);
        cache.advance(3);

        CHECK_EQ(cache.used(), size_t{3});
        CHECK_EQ(cache.keys(0).shape(), Shape({1, HKV, 3, DH}));
        CHECK_EQ(cache.keys(0).at({0, 0, 1, 0}), k.at({0, 0, 1, 0}));
        CHECK_EQ(cache.values(1).at({0, 1, 2, 1}), v.at({0, 1, 2, 1}));

        // the valid prefix is a VIEW over the pre-allocated storage, not a copy
        CHECK(cache.keys(0).storage() == cache.keys(0).storage());
        CHECK_EQ(cache.keys(0).storage()->size(), size_t{1 * HKV * 16 * DH});

        // appending one more grows the prefix
        Tensor one{Shape({1, HKV, 1, DH})};
        one.data()[0] = 99.0f;
        cache.append(0, one, one);
        cache.append(1, one, one);
        cache.advance(1);
        CHECK_EQ(cache.used(), size_t{4});
        CHECK_EQ(cache.keys(0).at({0, 0, 3, 0}), 99.0f);

        // reset empties it without reallocating
        const auto* storage_before = cache.keys(0).storage().get();
        cache.reset();
        CHECK_EQ(cache.used(), size_t{0});
        CHECK(cache.keys(0).storage().get() == storage_before);
    }

    // overflow throws, naming the limit
    {
        KVCache cache(1, 1, HKV, 4, DH);
        Tensor big{Shape({1, HKV, 5, DH})};
        CHECK_THROWS_AS(cache.append(0, big, big), std::runtime_error);
        try
        {
            cache.append(0, big, big);
        }
        catch (const std::runtime_error& error)
        {
            const std::string message = error.what();
            CHECK(message.find("limit of 4") != std::string::npos);
            CHECK(message.find("5 positions") != std::string::npos);
        }

        CHECK_THROWS_AS(cache.append(3, big, big), std::out_of_range);
        CHECK_THROWS_AS(cache.keys(9), std::out_of_range);
        CHECK_THROWS_AS(KVCache(0, 1, 1, 4, 2), std::invalid_argument);
    }

    // *** THE IDENTITY: cached and uncached generation produce the same ids, token for token ***
    // Exact, not approximate: both paths do the same arithmetic in the same order.
    {
        for (const std::vector<int32_t>& prompt :
             std::vector<std::vector<int32_t>>{{3}, {1, 2}, {5, 0, 7, 2}, {4, 4, 4, 4, 4, 4}})
        {
            GreedySampler greedy_uncached;
            GreedySampler greedy_cached;
            const Generator uncached(transformer, head, greedy_uncached);
            const Generator cached(transformer, head, greedy_cached);

            KVCache cache(LAYERS, 1, HKV, 64, DH);

            const Result plain = uncached.generate(prompt, {8, {}});
            const Result quick = cached.generate_cached(prompt, {8, {}}, cache);

            CHECK_EQ(plain.generated, quick.generated);
            CHECK_EQ(quick.generated.size(), size_t{8});
        }
    }

    // the same, with a stochastic sampler and a fixed seed
    {
        const std::vector<int32_t> prompt = {2, 6, 1};

        TemperatureSampler a(0.8f, 31337);
        TemperatureSampler b(0.8f, 31337);
        const Generator uncached(transformer, head, a);
        const Generator cached(transformer, head, b);

        KVCache cache(LAYERS, 1, HKV, 64, DH);
        const Result plain = uncached.generate(prompt, {10, {}});
        const Result quick = cached.generate_cached(prompt, {10, {}}, cache);

        CHECK_EQ(plain.generated, quick.generated);
    }

    // *** and the same hidden states, not just the same argmax ***
    {
        const std::vector<int64_t> prompt = {3, 1, 4, 1};

        const Tensor whole = transformer.forward(prompt, Shape({1, prompt.size()}));

        KVCache cache(LAYERS, 1, HKV, 64, DH);
        const Tensor prefilled = transformer.forward_cached(prompt, Shape({1, prompt.size()}), cache);

        CHECK_EQ(whole.shape(), prefilled.shape());
        for (size_t t = 0; t < prompt.size(); ++t)
        {
            for (size_t d = 0; d < D; ++d)
            {
                CHECK_NEAR(whole.at({0, t, d}), prefilled.at({0, t, d}), 1e-5);
            }
        }
        CHECK_EQ(cache.used(), prompt.size());

        // one more token, fed alone against the cache, matches the full run's last position
        const std::vector<int64_t> extended = {3, 1, 4, 1, 5};
        const Tensor full_again = transformer.forward(extended, Shape({1, extended.size()}));
        const Tensor stepped = transformer.forward_cached({5}, Shape({1, 1}), cache);

        CHECK_EQ(stepped.shape(), Shape({1, 1, D}));
        CHECK_EQ(cache.used(), size_t{5});
        for (size_t d = 0; d < D; ++d)
        {
            CHECK_NEAR(stepped.at({0, 0, d}), full_again.at({0, 4, d}), 1e-4);
        }
    }

    // the forward-pass economics: one pass per step either way, but each cached pass is one position
    {
        const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        GreedySampler sampler;
        const Generator generator(transformer, head, sampler);
        KVCache cache(LAYERS, 1, HKV, 64, DH);

        const Result quick = generator.generate_cached(prompt, {6, {}}, cache);

        CHECK_EQ(quick.generated.size(), size_t{6});
        CHECK_EQ(quick.forward_passes, size_t{7});      // one prefill plus six steps
        CHECK_EQ(cache.used(), prompt.size() + 6);      // and the cache holds every position once
    }

    // stop tokens and limits behave as they do without a cache
    {
        const std::vector<int32_t> prompt = {7};
        GreedySampler uncached_sampler;
        GreedySampler cached_sampler;
        const Generator uncached(transformer, head, uncached_sampler);
        const Generator cached(transformer, head, cached_sampler);

        const Result plain = uncached.generate(prompt, {5, {}});
        KVCache cache(LAYERS, 1, HKV, 64, DH);
        const Result quick = cached.generate_cached(prompt, {5, {}}, cache);
        CHECK_EQ(plain.generated, quick.generated);

        // a stop token that the model does produce ends both the same way
        const int32_t first = plain.generated.front();
        GreedySampler s1;
        GreedySampler s2;
        const Generator u2(transformer, head, s1);
        const Generator c2(transformer, head, s2);
        KVCache cache2(LAYERS, 1, HKV, 64, DH);

        const Result stopped_plain = u2.generate(prompt, {5, {first}});
        const Result stopped_quick = c2.generate_cached(prompt, {5, {first}}, cache2);
        CHECK_EQ(stopped_plain.generated, stopped_quick.generated);
        CHECK_EQ(stopped_plain.stopped_on_token, stopped_quick.stopped_on_token);
        CHECK(stopped_quick.stopped_on_token);

        // a cache can be reused for the next prompt
        cache2.reset();
        CHECK_EQ(c2.generate_cached({1, 1}, {3, {}}, cache2).generated.size(), size_t{3});
    }

    return VEDA_TEST_SUMMARY("KVCacheTest");
}
