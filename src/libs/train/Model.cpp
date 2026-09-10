#include "Model.h"

#include "Rope.h"
#include "Storage.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace veda::train
{

using autograd::Variable;
using core::Shape;
using core::Tensor;

namespace
{

// splitmix64 again: reproducible initial weights, because a run that cannot be repeated cannot be
// debugged.
struct Random
{
    uint64_t state;

    float uniform(float spread)
    {
        state += 0x9E3779B97F4A7C15ull;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        const double unit = static_cast<double>(z >> 11) / 9007199254740992.0;   // [0, 1)
        return static_cast<float>((unit * 2.0 - 1.0) * spread);
    }
};

// Initialisation is not decoration. Two failure modes bracket the choice: too large and the
// residual stream grows layer over layer until the softmax saturates and gradients vanish; too
// small and every head computes nearly the same thing and the model never differentiates them.
// 1/sqrt(fan_in) keeps the variance of a matmul's output roughly equal to its input's.
Tensor scaled(const Shape& shape, size_t fan_in, Random& random)
{
    const float spread = 1.0f / std::sqrt(static_cast<float>(fan_in));
    Tensor t{shape};
    for (size_t i = 0; i < t.numel(); ++i)
    {
        t.data()[i] = random.uniform(spread);
    }
    return t;
}

// A norm's weight is a per-channel gain, and 1 is the identity. Starting it at random would
// multiply the residual stream by noise before any learning happened.
Tensor ones(const Shape& shape)
{
    Tensor t{shape};
    for (size_t i = 0; i < t.numel(); ++i)
    {
        t.data()[i] = 1.0f;
    }
    return t;
}

std::string layer_name(size_t layer, const std::string& suffix)
{
    return "model.layers." + std::to_string(layer) + "." + suffix + ".weight";
}

} // namespace

io::ModelConfig tiny_config()
{
    io::ModelConfig config;
    config.hidden_size = 128;
    config.intermediate_size = 512;
    config.num_layers = 4;
    config.num_heads = 4;
    // No GQA: it exists to shrink the KV cache during generation, and training has no cache. It
    // would only cost a repeat operation with a gradient. Hkv == H is a perfectly valid file.
    config.num_kv_heads = 4;
    config.head_dim = 32;
    config.vocab_size = 256;           // byte-level: every string encodes, nothing is unknown
    config.max_position_embeddings = 256;
    config.rms_norm_eps = 1e-6f;
    config.rope_theta = 10000.0f;
    config.tie_word_embeddings = true;   // the LM head reuses the embedding matrix
    return config;
}

TinyTransformer::TinyTransformer(io::ModelConfig config, uint64_t seed)
    : config_(std::move(config)),
      embedding_(Variable::leaf(Tensor{Shape({1})})),
      final_norm_(Variable::leaf(Tensor{Shape({1})}))
{
    if (config_.num_heads == 0 || config_.num_kv_heads != config_.num_heads)
    {
        throw std::invalid_argument("TinyTransformer: training runs without GQA, so Hkv must be H");
    }
    if (!config_.tie_word_embeddings)
    {
        throw std::invalid_argument("TinyTransformer: only tied embeddings are supported");
    }

    Random random{seed == 0 ? 0x9E3779B97F4A7C15ull : seed};

    const size_t D = config_.hidden_size;
    const size_t F = config_.intermediate_size;
    const size_t H = config_.num_heads;
    const size_t Dh = config_.head_dim;
    const size_t V = config_.vocab_size;

    auto add = [&](const std::string& name, Tensor value) {
        Variable parameter = Variable::leaf(std::move(value));
        parameters_.push_back(parameter);
        names_.push_back(name);
        return parameter;
    };

    embedding_ = add("model.embed_tokens.weight", scaled(Shape({V, D}), D, random));

    layers_.reserve(config_.num_layers);
    for (size_t layer = 0; layer < config_.num_layers; ++layer)
    {
        Layer built{
            add(layer_name(layer, "input_layernorm"), ones(Shape({D}))),
            add(layer_name(layer, "self_attn.q_proj"), scaled(Shape({H * Dh, D}), D, random)),
            add(layer_name(layer, "self_attn.k_proj"), scaled(Shape({H * Dh, D}), D, random)),
            add(layer_name(layer, "self_attn.v_proj"), scaled(Shape({H * Dh, D}), D, random)),
            add(layer_name(layer, "self_attn.o_proj"), scaled(Shape({D, H * Dh}), H * Dh, random)),
            add(layer_name(layer, "self_attn.q_norm"), ones(Shape({Dh}))),
            add(layer_name(layer, "self_attn.k_norm"), ones(Shape({Dh}))),
            add(layer_name(layer, "post_attention_layernorm"), ones(Shape({D}))),
            add(layer_name(layer, "mlp.gate_proj"), scaled(Shape({F, D}), D, random)),
            add(layer_name(layer, "mlp.up_proj"), scaled(Shape({F, D}), D, random)),
            add(layer_name(layer, "mlp.down_proj"), scaled(Shape({D, F}), F, random))};
        layers_.push_back(std::move(built));
    }

    final_norm_ = add("model.norm.weight", ones(Shape({D})));
}

std::vector<std::pair<std::string, Tensor>> TinyTransformer::named_parameters() const
{
    std::vector<std::pair<std::string, Tensor>> named;
    named.reserve(parameters_.size());
    for (size_t i = 0; i < parameters_.size(); ++i)
    {
        named.emplace_back(names_[i], parameters_[i].value());
    }
    return named;
}

Variable TinyTransformer::attention(const Layer& layer, const Variable& x, size_t batch,
                                    size_t length, const Variable& mask, const Tensor& cosines,
                                    const Tensor& sines)
{
    const size_t H = config_.num_heads;
    const size_t Dh = config_.head_dim;

    // [B, T, H*Dh] -> [B, T, H, Dh] -> [B, H, T, Dh]
    auto split = [&](const Variable& projected) {
        return autograd::transpose(
            autograd::reshape(projected, Shape({batch, length, H, Dh})), 1, 2);
    };

    Variable q = split(autograd::matmul_nt(x, layer.q_proj));
    Variable k = split(autograd::matmul_nt(x, layer.k_proj));
    Variable v = split(autograd::matmul_nt(x, layer.v_proj));

    // QK-Norm BEFORE RoPE (E8): normalising after rotating would destroy the phase RoPE just
    // encoded.
    q = autograd::rope(
        autograd::mul(autograd::rms_normalize(q, config_.rms_norm_eps), layer.q_norm), cosines,
        sines);
    k = autograd::rope(
        autograd::mul(autograd::rms_normalize(k, config_.rms_norm_eps), layer.k_norm), cosines,
        sines);

    // Q . K^T, scaled by 1/sqrt(Dh) so the scores' variance does not grow with the head dimension
    // and push softmax into saturation.
    Variable scores = autograd::mul(autograd::matmul_nt(q, k),
                                    1.0f / std::sqrt(static_cast<float>(Dh)));

    // The mask is a constant leaf: 0 where a position may attend, -inf where it may not. softmax
    // turns -inf into exactly zero, so no gradient flows through a masked position either.
    scores = autograd::add(scores, mask);

    Variable context = autograd::matmul(autograd::softmax(scores, 3), v);

    // [B, H, T, Dh] -> [B, T, H, Dh] -> [B, T, H*Dh]
    context = autograd::reshape(autograd::transpose(context, 1, 2),
                                Shape({batch, length, H * Dh}));

    return autograd::matmul_nt(context, layer.o_proj);
}

Variable TinyTransformer::feed_forward(const Layer& layer, const Variable& x)
{
    // SwiGLU: one projection decides how much of the other passes.
    Variable gated = autograd::mul(autograd::silu(autograd::matmul_nt(x, layer.gate)),
                                   autograd::matmul_nt(x, layer.up));
    return autograd::matmul_nt(gated, layer.down);
}

Variable TinyTransformer::forward(const std::vector<int64_t>& ids, size_t batch, size_t length)
{
    if (ids.size() != batch * length)
    {
        throw std::invalid_argument("TinyTransformer::forward: " + std::to_string(ids.size()) +
                                    " ids for a [" + std::to_string(batch) + ", " +
                                    std::to_string(length) + "] window");
    }
    if (length > config_.max_position_embeddings)
    {
        throw std::invalid_argument("TinyTransformer::forward: window of " +
                                    std::to_string(length) + " exceeds the context limit " +
                                    std::to_string(config_.max_position_embeddings));
    }

    // Built once per forward, not per layer: it depends only on T.
    Tensor mask_value{Shape({1, 1, length, length})};
    for (size_t t = 0; t < length; ++t)
    {
        for (size_t s = 0; s < length; ++s)
        {
            mask_value.data()[t * length + s] =
                s <= t ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }
    const Variable mask = Variable::leaf(std::move(mask_value), false);

    const model::RopeTables tables =
        model::rope_tables(length, config_.head_dim, config_.rope_theta);

    Variable x = autograd::embedding(embedding_, ids, Shape({batch, length}));

    for (const Layer& layer : layers_)
    {
        // Pre-norm residual: the stream itself is never normalised, only what each sublayer reads
        // from it — which is why a 28-layer stack trains at all.
        Variable normed =
            autograd::mul(autograd::rms_normalize(x, config_.rms_norm_eps), layer.input_norm);
        x = autograd::add(x, attention(layer, normed, batch, length, mask, tables.cosines,
                                       tables.sines));

        normed = autograd::mul(autograd::rms_normalize(x, config_.rms_norm_eps), layer.post_norm);
        x = autograd::add(x, feed_forward(layer, normed));
    }

    x = autograd::mul(autograd::rms_normalize(x, config_.rms_norm_eps), final_norm_);

    // Tied: the LM head is the embedding matrix read the other way round.
    return autograd::matmul_nt(x, embedding_);
}

} // namespace veda::train
