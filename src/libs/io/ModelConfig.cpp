#include "ModelConfig.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace veda::io
{

namespace
{

[[noreturn]] void fail(const std::string& reason)
{
    throw std::runtime_error("model config: " + reason);
}

size_t required_size(const JsonValue& document, const std::string& field)
{
    if (!document.contains(field))
    {
        fail("missing \"" + field + "\"");
    }
    const int64_t value = document[field].as_int();   // the JSON reader rejects a non-integral one
    if (value <= 0)
    {
        fail("\"" + field + "\" is " + std::to_string(value) + ", must be positive");
    }
    return static_cast<size_t>(value);
}

float required_float(const JsonValue& document, const std::string& field)
{
    if (!document.contains(field))
    {
        fail("missing \"" + field + "\"");
    }
    const double value = document[field].as_double();
    if (!(value > 0.0))
    {
        fail("\"" + field + "\" is " + std::to_string(value) + ", must be positive");
    }
    return static_cast<float>(value);
}

} // namespace

ModelConfig parse_model_config(const JsonValue& document)
{
    if (!document.is_object())
    {
        fail("the document is not a JSON object");
    }

    ModelConfig config;
    config.hidden_size = required_size(document, "hidden_size");
    config.intermediate_size = required_size(document, "intermediate_size");
    config.num_layers = required_size(document, "num_hidden_layers");
    config.num_heads = required_size(document, "num_attention_heads");
    config.num_kv_heads = required_size(document, "num_key_value_heads");
    config.vocab_size = required_size(document, "vocab_size");
    config.rms_norm_eps = required_float(document, "rms_norm_eps");
    config.rope_theta = required_float(document, "rope_theta");

    // Stated in the file when present, and it is not obliged to equal D/H — when the two disagree,
    // the file is right and the assumption is wrong. Every projection's output width follows from
    // this: q_proj produces H*Dh, k_proj and v_proj produce Hkv*Dh, and neither is generally D.
    if (document.contains("head_dim"))
    {
        config.head_dim = required_size(document, "head_dim");
    }
    else
    {
        if (config.hidden_size % config.num_heads != 0)
        {
            fail("\"head_dim\" is absent and hidden_size " + std::to_string(config.hidden_size) +
                 " is not divisible by num_attention_heads " + std::to_string(config.num_heads));
        }
        config.head_dim = config.hidden_size / config.num_heads;
    }

    // Not every export writes these two, and both have a defensible fallback — unlike a dimension,
    // which never gets one.
    config.max_position_embeddings = document.contains("max_position_embeddings")
                                         ? required_size(document, "max_position_embeddings")
                                         : 0;
    config.tie_word_embeddings =
        document.contains("tie_word_embeddings") && document["tie_word_embeddings"].as_bool();

    // GQA: several query heads share one key/value head, so the ratio must divide exactly. A config
    // where it does not would let E8's expansion mis-pair heads silently.
    if (config.num_kv_heads > config.num_heads)
    {
        fail("num_key_value_heads " + std::to_string(config.num_kv_heads) +
             " exceeds num_attention_heads " + std::to_string(config.num_heads));
    }
    if (config.num_heads % config.num_kv_heads != 0)
    {
        fail("num_attention_heads " + std::to_string(config.num_heads) +
             " does not divide into num_key_value_heads " + std::to_string(config.num_kv_heads));
    }

    return config;
}

ModelConfig read_model_config(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("model config \"" + path + "\": cannot be opened");
    }

    std::ostringstream buffer;
    buffer << in.rdbuf();
    return parse_model_config(parse_json(buffer.str()));
}

std::string ModelConfig::to_string() const
{
    std::ostringstream out;
    out << "ModelConfig\n"
        << "  layers            " << num_layers << "\n"
        << "  hidden size   D   " << hidden_size << "\n"
        << "  ffn inner     F   " << intermediate_size << "\n"
        << "  query heads   H   " << num_heads << "\n"
        << "  kv heads      Hkv " << num_kv_heads << "  (" << queries_per_kv_head()
        << " queries per kv head)\n"
        << "  head dim      Dh  " << head_dim << "\n"
        << "  vocabulary    V   " << vocab_size << "\n"
        << "  rms norm eps      " << rms_norm_eps << "\n"
        << "  rope theta        " << rope_theta << "\n"
        << "  context limit     " << max_position_embeddings << "\n"
        << "  tied embeddings   " << (tie_word_embeddings ? "yes" : "no");
    return out.str();
}

} // namespace veda::io
