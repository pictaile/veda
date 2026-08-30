// E4.S3.T5 (the hyperparameters) and T6 (ModelConfig)

#include "Json.h"
#include "ModelConfig.h"
#include "TestSupport.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

using veda::io::ModelConfig;
using veda::io::JsonValue;
using veda::io::parse_json;
using veda::io::parse_model_config;
using veda::io::read_model_config;

namespace
{
// A Qwen3-shaped config, with the extra fields a real export carries.
const char* full_config = R"({
  "architectures": ["Qwen3ForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 40960,
  "max_window_layers": 28,
  "model_type": "qwen3",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "use_cache": true,
  "vocab_size": 151936
})";

std::string message_of(const std::string& json)
{
    try
    {
        (void)parse_model_config(parse_json(json));
    }
    catch (const std::runtime_error& error)
    {
        return error.what();
    }
    return "";
}

// A config with only what is required, for the negative cases.
std::string minimal(const std::string& overrides = "")
{
    return std::string(R"({"hidden_size": 1024, "intermediate_size": 3072,)") +
           R"("num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,)" +
           R"("vocab_size": 151936, "rms_norm_eps": 1e-06, "rope_theta": 1000000.0)" + overrides +
           "}";
}
} // namespace

int main()
{
    // the full config parses, and every field reads back
    {
        const ModelConfig config = parse_model_config(parse_json(full_config));

        CHECK_EQ(config.hidden_size, size_t{1024});
        CHECK_EQ(config.intermediate_size, size_t{3072});
        CHECK_EQ(config.num_layers, size_t{28});
        CHECK_EQ(config.num_heads, size_t{16});
        CHECK_EQ(config.num_kv_heads, size_t{8});
        CHECK_EQ(config.head_dim, size_t{128});
        CHECK_EQ(config.vocab_size, size_t{151936});
        CHECK_EQ(config.max_position_embeddings, size_t{40960});
        CHECK_NEAR(config.rms_norm_eps, 1e-6, 1e-12);
        CHECK_NEAR(config.rope_theta, 1000000.0, 1.0);
        CHECK(config.tie_word_embeddings);

        // GQA: each kv head serves two query heads, which halves the cache in E12
        CHECK_EQ(config.queries_per_kv_head(), size_t{2});

        // the thirty-odd fields Veda does not read are simply ignored
        CHECK_EQ(config.hidden_size, size_t{1024});
    }

    // head_dim is what the file says, not D/H — the two need not agree
    {
        const ModelConfig stated = parse_model_config(parse_json(full_config));
        CHECK_EQ(stated.head_dim, size_t{128});
        CHECK(stated.head_dim != stated.hidden_size / stated.num_heads);   // 128 != 64

        // absent: D/H is the fallback, and then divisibility becomes a requirement
        const ModelConfig derived = parse_model_config(parse_json(minimal()));
        CHECK_EQ(derived.head_dim, size_t{64});
        CHECK_EQ(derived.head_dim, derived.hidden_size / derived.num_heads);

        CHECK(message_of(R"({"hidden_size": 100, "intermediate_size": 300,
            "num_hidden_layers": 2, "num_attention_heads": 16, "num_key_value_heads": 8,
            "vocab_size": 100, "rms_norm_eps": 1e-06, "rope_theta": 10000.0})")
                  .find("not divisible") != std::string::npos);
    }

    // the shapes the rest of the runtime derives from these numbers
    {
        const ModelConfig config = parse_model_config(parse_json(full_config));

        const size_t q_out = config.num_heads * config.head_dim;      // [H*Dh, D]
        const size_t kv_out = config.num_kv_heads * config.head_dim;  // [Hkv*Dh, D]
        CHECK_EQ(q_out, size_t{2048});
        CHECK_EQ(kv_out, size_t{1024});
        CHECK(q_out != config.hidden_size);   // and it is not D — the assumption that would bite
        CHECK_EQ(q_out / kv_out, config.queries_per_kv_head());
    }

    // missing required fields throw, naming the field
    {
        CHECK_THROWS_AS(parse_model_config(parse_json(R"({"hidden_size": 1024})")),
                        std::runtime_error);
        CHECK(message_of(R"({"hidden_size": 1024})").find("intermediate_size") != std::string::npos);

        for (const char* field : {"hidden_size", "intermediate_size", "num_hidden_layers",
                                  "num_attention_heads", "num_key_value_heads", "vocab_size",
                                  "rms_norm_eps", "rope_theta"})
        {
            // build the minimal config without this one field
            const JsonValue full = parse_json(minimal());
            std::string reduced = "{";
            for (const auto& member : full.members())
            {
                if (member.first == field)
                {
                    continue;
                }
                if (reduced.size() > 1)
                {
                    reduced += ",";
                }
                reduced += "\"" + member.first + "\": " +
                           (member.second.is_number() ? std::to_string(member.second.as_double())
                                                      : "0");
            }
            reduced += "}";
            const std::string message = message_of(reduced);
            CHECK(message.find(field) != std::string::npos);
        }
    }

    // the GQA relations
    {
        const std::string uneven =
            R"({"hidden_size": 1024, "intermediate_size": 3072, "num_hidden_layers": 28,
                "num_attention_heads": 17, "num_key_value_heads": 8, "vocab_size": 100,
                "rms_norm_eps": 1e-06, "rope_theta": 10000.0, "head_dim": 64})";
        CHECK(message_of(uneven).find("does not divide") != std::string::npos);
        CHECK(message_of(uneven).find("17") != std::string::npos);

        const std::string too_many_kv =
            R"({"hidden_size": 1024, "intermediate_size": 3072, "num_hidden_layers": 28,
                "num_attention_heads": 8, "num_key_value_heads": 16, "vocab_size": 100,
                "rms_norm_eps": 1e-06, "rope_theta": 10000.0, "head_dim": 64})";
        CHECK(message_of(too_many_kv).find("exceeds") != std::string::npos);

        // H == Hkv is plain multi-head attention, and perfectly legal
        const std::string no_grouping =
            R"({"hidden_size": 1024, "intermediate_size": 3072, "num_hidden_layers": 28,
                "num_attention_heads": 16, "num_key_value_heads": 16, "vocab_size": 100,
                "rms_norm_eps": 1e-06, "rope_theta": 10000.0, "head_dim": 64})";
        CHECK_EQ(parse_model_config(parse_json(no_grouping)).queries_per_kv_head(), size_t{1});
    }

    // zeros and non-numbers are refused
    {
        CHECK(message_of(R"({"hidden_size": 0, "intermediate_size": 3072, "num_hidden_layers": 28,
            "num_attention_heads": 16, "num_key_value_heads": 8, "vocab_size": 100,
            "rms_norm_eps": 1e-06, "rope_theta": 10000.0})")
                  .find("must be positive") != std::string::npos);

        CHECK(message_of(R"({"hidden_size": 1024, "intermediate_size": 3072, "num_hidden_layers": 0,
            "num_attention_heads": 16, "num_key_value_heads": 8, "vocab_size": 100,
            "rms_norm_eps": 1e-06, "rope_theta": 10000.0})")
                  .find("num_hidden_layers") != std::string::npos);

        // eps of zero would let an all-zero vector divide by zero in RMSNorm
        CHECK(message_of(R"({"hidden_size": 1024, "intermediate_size": 3072, "num_hidden_layers": 28,
            "num_attention_heads": 16, "num_key_value_heads": 8, "vocab_size": 100,
            "rms_norm_eps": 0, "rope_theta": 10000.0})")
                  .find("rms_norm_eps") != std::string::npos);

        // a non-integral dimension is caught by the JSON reader itself
        CHECK_THROWS_AS(parse_model_config(parse_json(
                            R"({"hidden_size": 1.5, "intermediate_size": 3072,
                                "num_hidden_layers": 28, "num_attention_heads": 16,
                                "num_key_value_heads": 8, "vocab_size": 100,
                                "rms_norm_eps": 1e-06, "rope_theta": 10000.0})")),
                        std::runtime_error);

        CHECK_THROWS_AS(parse_model_config(parse_json("[1, 2]")), std::runtime_error);
    }

    // optional fields fall back; dimensions never do
    {
        const ModelConfig config = parse_model_config(parse_json(minimal()));
        CHECK_EQ(config.max_position_embeddings, size_t{0});   // absent
        CHECK(!config.tie_word_embeddings);                    // absent means false

        const ModelConfig tied = parse_model_config(parse_json(minimal(R"(, "tie_word_embeddings": false)")));
        CHECK(!tied.tie_word_embeddings);
    }

    // reading from a file
    {
        const auto path = std::filesystem::temp_directory_path() / "veda_config.json";
        {
            std::ofstream out(path);
            out << full_config;
        }
        const ModelConfig config = read_model_config(path.string());
        CHECK_EQ(config.num_layers, size_t{28});
        std::filesystem::remove(path);

        CHECK_THROWS_AS(read_model_config((path / "absent.json").string()), std::runtime_error);
    }

    // to_string renders the dimensions
    {
        const std::string rendered = parse_model_config(parse_json(full_config)).to_string();
        CHECK(rendered.find("1024") != std::string::npos);
        CHECK(rendered.find("28") != std::string::npos);
        CHECK(rendered.find("queries per kv head") != std::string::npos);
        CHECK(rendered.find("tied embeddings   yes") != std::string::npos);
    }

    return VEDA_TEST_SUMMARY("ModelConfigTest");
}
