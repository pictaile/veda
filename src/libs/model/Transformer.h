#ifndef VEDA_TRANSFORMER_H
#define VEDA_TRANSFORMER_H

#include "Embedding.h"
#include "KVCache.h"
#include "RMSNorm.h"
#include "Shape.h"
#include "Tensor.h"
#include "TransformerBlock.h"

#include <cstdint>
#include <vector>

namespace veda::model
{

// The stack: embedding, N blocks, final norm.
//
// The tensor flowing through the loop is the residual stream — one [B, T, D] tensor that every
// block reads and adds to. "The output of layer 7" is really "everything the model has accumulated
// by layer 7": because blocks add rather than replace (E9.S2.T3), what layer 2 wrote is still there
// at layer 27 unless something removed it.
//
// After the last block the stream is un-normalised and has grown across 28 layers of additions, so
// a final RMSNorm sits between it and the LM head.
class Transformer
{
public:
    struct Weights
    {
        nn::Embedding embedding;
        std::vector<TransformerBlock> blocks;
        nn::RMSNorm final_norm;
    };

    explicit Transformer(Weights weights);

    // [B, T] ids -> [B, T, D]
    core::Tensor forward(const std::vector<int64_t>& ids, const core::Shape& id_shape,
                         size_t first_position = 0) const;

    // Every intermediate, for comparing against a reference layer by layer.
    //
    // With 28 layers, "the output is wrong" is not a debuggable statement; with per-layer captures
    // it becomes a binary search, and five comparisons find the first divergent layer. That is what
    // E3's harness was built for, and this is where it plugs in.
    //
    // Costs 28 * T * D * 4 bytes — 3.6 MB at T = 32 — which is why it is a separate method rather
    // than an always-on side effect.
    struct Trace
    {
        core::Tensor embeddings;
        std::vector<core::Tensor> layer_outputs;
        core::Tensor final_hidden;
    };

    Trace forward_capturing(const std::vector<int64_t>& ids, const core::Shape& id_shape,
                            size_t first_position = 0) const;

    // With a cache, every step processes only the new positions: prefill passes the whole prompt,
    // and each generation step passes one token. The cache advances once, after every layer has
    // appended at the same offset.
    core::Tensor forward_cached(const std::vector<int64_t>& ids, const core::Shape& id_shape,
                                KVCache& cache) const;

    const nn::Embedding& embedding() const noexcept { return weights_.embedding; }
    const std::vector<TransformerBlock>& blocks() const noexcept { return weights_.blocks; }
    const nn::RMSNorm& final_norm() const noexcept { return weights_.final_norm; }
    size_t hidden_size() const { return weights_.embedding.hidden_size(); }

private:
    Weights weights_;
};

} // namespace veda::model

#endif //VEDA_TRANSFORMER_H
