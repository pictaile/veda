#ifndef VEDA_TRAIN_MODEL_H
#define VEDA_TRAIN_MODEL_H

#include "ModelConfig.h"
#include "Shape.h"
#include "Tensor.h"
#include "Variable.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// The same architecture as the inference model, on the tape.
//
// The inference Transformer (E9) computes on core::Tensor, one position at a time, through a KV
// cache — and it is the object E3's reference comparison certifies. Training needs the same
// computation on autograd::Variable, over a whole [B, T] window at once, with no cache.
//
// Making one class serve both would mean templating the entire model stack over its value type, and
// would put the verified inference path at risk to avoid writing a forward pass twice. THE
// DUPLICATION IS THE CHEAPER MISTAKE, and it is bounded: what this shares with inference is the
// thing that actually matters — the parameter layout.
//
// Every weight is named and shaped exactly as E4's loader expects, so a checkpoint written here is
// a file veda_run can generate from with no special case. That is the epic's milestone, and it is a
// property of naming rather than of code sharing.
namespace veda::train
{

// A ModelConfig sized for training from scratch: ~1 M parameters instead of 596 M.
io::ModelConfig tiny_config();

class TinyTransformer
{
public:
    TinyTransformer(io::ModelConfig config, uint64_t seed);

    // ids are B*T token ids; the result is logits [B, T, V] on the tape.
    autograd::Variable forward(const std::vector<int64_t>& ids, size_t batch, size_t length);

    // Every trainable leaf, once each, in a stable order.
    const std::vector<autograd::Variable>& parameters() const noexcept { return parameters_; }

    // The same tensors under E4's names — the contract with the loader.
    std::vector<std::pair<std::string, core::Tensor>> named_parameters() const;

    const io::ModelConfig& config() const noexcept { return config_; }

private:
    struct Layer
    {
        autograd::Variable input_norm;
        autograd::Variable q_proj, k_proj, v_proj, o_proj;
        autograd::Variable q_norm, k_norm;
        autograd::Variable post_norm;
        autograd::Variable gate, up, down;
    };

    io::ModelConfig config_;
    autograd::Variable embedding_;
    std::vector<Layer> layers_;
    autograd::Variable final_norm_;

    std::vector<autograd::Variable> parameters_;
    std::vector<std::string> names_;

    autograd::Variable attention(const Layer& layer, const autograd::Variable& x, size_t batch,
                                 size_t length, const autograd::Variable& mask,
                                 const core::Tensor& cosines, const core::Tensor& sines);
    autograd::Variable feed_forward(const Layer& layer, const autograd::Variable& x);
};

} // namespace veda::train

#endif //VEDA_TRAIN_MODEL_H
