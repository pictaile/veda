#ifndef VEDA_MODEL_CONFIG_H
#define VEDA_MODEL_CONFIG_H

#include "Json.h"

#include <cstddef>
#include <string>

namespace veda::io
{

// The dimensions of the model, read from config.json.
//
// Eleven numbers decide the shape of every tensor in the runtime, and every one of them appears in
// an assertion somewhere in E6 through E10. They are read, never hard-coded: that is what makes
// Qwen3-1.7B reachable by swapping a directory (AD9), and it is an acceptance criterion of E9 and
// E10.
//
// Validated once, here, so that everything above may assume: H % Hkv == 0 checked at load is why
// E8's head expansion can assert rather than handle.
struct ModelConfig
{
    size_t hidden_size = 0;              // D — the width of the residual stream
    size_t intermediate_size = 0;        // F — the FFN's inner width
    size_t num_layers = 0;
    size_t num_heads = 0;                // H
    size_t num_kv_heads = 0;             // Hkv — fewer than H, which is GQA
    size_t head_dim = 0;                 // Dh — stated in the file, or D/H when absent
    size_t vocab_size = 0;               // V
    size_t max_position_embeddings = 0;  // the context limit, and the KV cache size (E12)
    float rms_norm_eps = 0.0f;           // the guard inside every RMSNorm's square root
    float rope_theta = 0.0f;             // the base of RoPE's frequency ladder
    bool tie_word_embeddings = false;    // true: the LM head reuses the embedding matrix

    // The GQA group size: how many query heads share one key/value head.
    size_t queries_per_kv_head() const { return num_heads / num_kv_heads; }

    std::string to_string() const;
};

// Reads the eleven fields and ignores every other one — a real config.json has thirty-odd, and
// ignoring the rest is what lets a config from a newer export still load. Throws
// std::runtime_error naming the field for anything missing or invalid.
ModelConfig parse_model_config(const JsonValue& document);

ModelConfig read_model_config(const std::string& path);

} // namespace veda::io

#endif //VEDA_MODEL_CONFIG_H
