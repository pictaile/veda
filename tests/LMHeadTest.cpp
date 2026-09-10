// E10.S3.T4 — the LM head, and the MVP hook ★

#include "Binding.h"
#include "Compare.h"
#include "LMHead.h"
#include "ModelConfig.h"
#include "Safetensors.h"
#include "Shape.h"
#include "TensorFile.h"
#include "TestSupport.h"
#include "Tensor.h"
#include "Transformer.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::model::LMHead;

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

// The greedy choice. E11 turns this into GreedySampler; here it is what the MVP criterion needs.
size_t argmax(const Tensor& logits, size_t batch)
{
    size_t best = 0;
    for (size_t v = 1; v < logits.shape()[1]; ++v)
    {
        if (logits.at({batch, v}) > logits.at({batch, best}))
        {
            best = v;
        }
    }
    return best;
}
} // namespace

int main()
{
    // the hand-computed example: V = 3, D = 2, tied
    //   table  = [[1,0],[0,1],[1,1]]     hidden = [0.9, 0.1]
    //   logits = [0.9, 0.1, 1.0]         argmax = 2
    {
        const Tensor table = filled(Shape({3, 2}), {1, 0, 0, 1, 1, 1});
        const LMHead head(table);

        CHECK_EQ(head.vocab_size(), size_t{3});
        CHECK_EQ(head.hidden_size(), size_t{2});

        const Tensor hidden = filled(Shape({1, 1, 2}), {0.9f, 0.1f});
        const Tensor logits = head.forward(hidden);

        CHECK_EQ(logits.shape(), Shape({1, 1, 3}));
        CHECK_NEAR(logits.at({0, 0, 0}), 0.9, 1e-6);
        CHECK_NEAR(logits.at({0, 0, 1}), 0.1, 1e-6);
        CHECK_NEAR(logits.at({0, 0, 2}), 1.0, 1e-6);

        const Tensor last = head.forward_last(hidden);
        CHECK_EQ(last.shape(), Shape({1, 3}));
        CHECK_EQ(argmax(last, 0), size_t{2});

        // token 2 wins although the hidden state points most like token 0: its row is longer, and
        // a dot product rewards length as well as direction
        CHECK(logits.at({0, 0, 2}) > logits.at({0, 0, 0}));
    }

    // *** forward_last equals the last row of forward, exactly ***
    {
        const size_t V = 5;
        const size_t D = 3;
        Tensor table{Shape({V, D})};
        for (size_t i = 0; i < table.numel(); ++i)
        {
            table.data()[i] = 0.1f * static_cast<float>(i) - 0.7f;
        }
        const LMHead head(table);

        Tensor hidden{Shape({2, 4, D})};
        for (size_t i = 0; i < hidden.numel(); ++i)
        {
            hidden.data()[i] = 0.05f * static_cast<float>(i) - 0.3f;
        }

        const Tensor full = head.forward(hidden);
        const Tensor last = head.forward_last(hidden);

        CHECK_EQ(full.shape(), Shape({2, 4, V}));
        CHECK_EQ(last.shape(), Shape({2, V}));

        for (size_t b = 0; b < 2; ++b)
        {
            for (size_t v = 0; v < V; ++v)
            {
                CHECK_EQ(last.at({b, v}), full.at({b, 3, v}));
            }
        }

        // and the argmax agrees, which is the property the MVP rests on
        for (size_t b = 0; b < 2; ++b)
        {
            size_t best_full = 0;
            for (size_t v = 1; v < V; ++v)
            {
                best_full = full.at({b, 3, v}) > full.at({b, 3, best_full}) ? v : best_full;
            }
            CHECK_EQ(argmax(last, b), best_full);
        }
    }

    // what slicing first saves: the full head is T times the work of the last row
    {
        const size_t V = 151936;
        const size_t D = 1024;
        const size_t T = 8;
        CHECK_EQ(T * V * sizeof(float), size_t{4'861'952});   // 4.6 MB for every position
        CHECK_EQ(V * sizeof(float), size_t{607'744});         // 594 KB for the last one
        CHECK_EQ(V * D, size_t{155'582'464});                 // and a third of a token's arithmetic
    }

    // tied embeddings: a head built from the transformer's table gives the same logits as one
    // built from a copy — the point being that no second matrix exists
    {
        const Tensor table = filled(Shape({3, 2}), {1, 0, 0, 1, 1, 1});
        const LMHead from_table(table);
        const LMHead from_copy(veda::core::contiguous(table));

        const Tensor hidden = filled(Shape({1, 1, 2}), {0.3f, 0.7f});
        for (size_t v = 0; v < 3; ++v)
        {
            CHECK_EQ(from_table.forward(hidden).at({0, 0, v}),
                     from_copy.forward(hidden).at({0, 0, v}));
        }

        // and the weight is shared, not copied (AD2)
        CHECK(from_table.weight().storage() == table.storage());
        CHECK_EQ(table.storage().use_count(), long{2});
    }

    // refusals
    {
        const LMHead head(filled(Shape({3, 2}), {1, 0, 0, 1, 1, 1}));

        CHECK_THROWS_AS(LMHead(Tensor{Shape({3})}), std::invalid_argument);
        CHECK_THROWS_AS(LMHead(Tensor{Shape({1, 2, 3})}), std::invalid_argument);
        CHECK_THROWS_AS(head.forward(Tensor{Shape({1, 1, 5})}), std::invalid_argument);
        CHECK_THROWS_AS(head.forward_last(Tensor{Shape({1, 2})}), std::invalid_argument);
        CHECK_THROWS_AS(head.forward_last(Tensor{Shape({1, 0, 2})}), std::invalid_argument);

        // V = 1 is legal, if pointless
        const LMHead single(filled(Shape({1, 2}), {1, 1}));
        CHECK_EQ(single.forward_last(filled(Shape({1, 1, 2}), {2, 3})).at({0, 0}), 5.0f);
    }

    // --- the whole stack, end to end, on a synthetic model -----------------------------------------
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
            Tensor t{shape};
            for (size_t i = 0; i < t.numel(); ++i)
            {
                t.data()[i] = 0.01f * static_cast<float>((i * 7) % 23) - 0.1f;
            }
            weights.add(name, t, shape.size() * 4);
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

        const veda::model::Transformer transformer = veda::binding::bind(weights, config);
        // tied: the head is the embedding table itself
        const LMHead head(transformer.embedding().table());

        const std::vector<int64_t> ids = {3, 1, 4, 1, 5};
        const Tensor hidden = transformer.forward(ids, Shape({1, ids.size()}));
        CHECK_EQ(hidden.shape(), Shape({1, 5, 8}));

        const Tensor logits = head.forward_last(hidden);
        CHECK_EQ(logits.shape(), Shape({1, 16}));

        // every logit is finite, and one of them is the largest
        bool finite = true;
        for (size_t v = 0; v < 16; ++v)
        {
            if (!std::isfinite(logits.at({0, v})))
            {
                finite = false;
            }
        }
        CHECK(finite);

        const size_t next = argmax(logits, 0);
        CHECK(next < 16);

        // deterministic: the same prompt gives the same token, from a cold start
        const veda::model::Transformer again = veda::binding::bind(weights, config);
        const LMHead head_again(again.embedding().table());
        CHECK_EQ(argmax(head_again.forward_last(again.forward(ids, Shape({1, ids.size()}))), 0),
                 next);
    }

    // --- ★ THE MVP CRITERION: the top-1 token matches the reference --------------------------------
    {
        const std::filesystem::path reference = "reference/logits_last.bin";
        if (std::filesystem::exists(reference))
        {
            const Tensor expected = veda::testing::read_tensor(reference.string());
            std::cout << "  reference logits: " << expected.shape().to_string() << "\n";
            CHECK_EQ(expected.rank(), size_t{2});
            // The comparison itself needs the real weights: run tools/veda_run against the model
            // directory and compare its top-1 against this file's argmax.
        }
        else
        {
            std::cout << "  SKIPPED: no reference/logits_last.bin.\n"
                      << "  The MVP criterion — the same top-1 token as the reference — needs the\n"
                      << "  real weights: python3 tools/dump_reference.py, then ./veda_run.\n";
        }
    }

    return VEDA_TEST_SUMMARY("LMHeadTest");
}
