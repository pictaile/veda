#include "Binding.h"

#include "FeedForward.h"
#include "Linear.h"
#include "ScaledDotProductAttention.h"
#include "Shape.h"

#include <set>
#include <stdexcept>
#include <string>

namespace veda::binding
{

using core::Shape;
using core::Tensor;

namespace
{

class Taker
{
public:
    explicit Taker(const io::WeightFile& weights) : weights_(weights) {}

    // Asks for a name and records that it was taken. A tensor that exists with the wrong shape
    // means the file and the config describe different models.
    const Tensor& take(const std::string& name, const Shape& expected)
    {
        if (!weights_.contains(name))
        {
            throw std::runtime_error("binding: no tensor named \"" + name + "\"");
        }

        const Tensor& tensor = weights_[name];
        if (tensor.shape() != expected)
        {
            throw std::runtime_error("binding: \"" + name + "\" is " + tensor.shape().to_string() +
                                     ", the config implies " + expected.to_string());
        }

        taken_.insert(name);
        return tensor;
    }

    // The direction people forget: anything left in the file was ignored, which almost always means
    // a name was misspelled and a real weight was skipped in favour of whatever a layer defaulted to.
    void require_nothing_left() const
    {
        std::string first;
        size_t unused = 0;
        for (const std::string& name : weights_.names())
        {
            if (!taken_.contains(name))
            {
                ++unused;
                if (first.empty())
                {
                    first = name;
                }
            }
        }
        if (unused > 0)
        {
            throw std::runtime_error("binding: " + std::to_string(unused) +
                                     " tensor(s) in the file were not used, first \"" + first + "\"");
        }
    }

private:
    const io::WeightFile& weights_;
    std::set<std::string> taken_;
};

std::string layer_name(size_t layer, const std::string& suffix)
{
    return "model.layers." + std::to_string(layer) + "." + suffix + ".weight";
}

} // namespace

model::Transformer bind(const io::WeightFile& weights, const io::ModelConfig& config)
{
    Taker taker(weights);

    const size_t D = config.hidden_size;
    const size_t F = config.intermediate_size;
    const size_t Dh = config.head_dim;
    const size_t H = config.num_heads;
    const size_t Hkv = config.num_kv_heads;
    const size_t V = config.vocab_size;

    nn::Embedding embedding(taker.take("model.embed_tokens.weight", Shape({V, D})));

    std::vector<model::TransformerBlock> blocks;
    blocks.reserve(config.num_layers);

    for (size_t layer = 0; layer < config.num_layers; ++layer)
    {
        // QK-Norm is required, not optional-if-present: a Qwen3 file without q_norm is not a Qwen3
        // file. The optionality in the attention layer exists for tests and other model families,
        // not to make a missing weight silently acceptable here.
        model::ScaledDotProductAttention attention(
            {nn::Linear(taker.take(layer_name(layer, "self_attn.q_proj"), Shape({H * Dh, D}))),
             nn::Linear(taker.take(layer_name(layer, "self_attn.k_proj"), Shape({Hkv * Dh, D}))),
             nn::Linear(taker.take(layer_name(layer, "self_attn.v_proj"), Shape({Hkv * Dh, D}))),
             nn::Linear(taker.take(layer_name(layer, "self_attn.o_proj"), Shape({D, H * Dh}))),
             nn::RMSNorm(taker.take(layer_name(layer, "self_attn.q_norm"), Shape({Dh})),
                         config.rms_norm_eps),
             nn::RMSNorm(taker.take(layer_name(layer, "self_attn.k_norm"), Shape({Dh})),
                         config.rms_norm_eps)},
            {H, Hkv, Dh, config.rope_theta});

        model::FeedForward feed_forward(
            {nn::Linear(taker.take(layer_name(layer, "mlp.gate_proj"), Shape({F, D}))),
             nn::Linear(taker.take(layer_name(layer, "mlp.up_proj"), Shape({F, D}))),
             nn::Linear(taker.take(layer_name(layer, "mlp.down_proj"), Shape({D, F})))});

        blocks.emplace_back(model::TransformerBlock::Weights{
            nn::RMSNorm(taker.take(layer_name(layer, "input_layernorm"), Shape({D})),
                        config.rms_norm_eps),
            std::move(attention),
            nn::RMSNorm(taker.take(layer_name(layer, "post_attention_layernorm"), Shape({D})),
                        config.rms_norm_eps),
            std::move(feed_forward)});
    }

    nn::RMSNorm final_norm(taker.take("model.norm.weight", Shape({D})), config.rms_norm_eps);

    // The LM head has no weights of its own when the embeddings are tied (§4 of the architecture) —
    // looking for lm_head.weight in such a file fails, because it is not there. When they are not
    // tied it exists, and taking it here keeps the both-directions check honest.
    if (!config.tie_word_embeddings)
    {
        (void)taker.take("lm_head.weight", Shape({V, D}));
    }

    taker.require_nothing_left();

    return model::Transformer(
        {std::move(embedding), std::move(blocks), std::move(final_norm)});
}

} // namespace veda::binding
