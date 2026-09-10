#ifndef VEDA_KV_CACHE_H
#define VEDA_KV_CACHE_H

#include "Tensor.h"

#include <cstddef>
#include <vector>

namespace veda::model
{

// Keys and values of earlier positions do not change, so compute them once.
//
// Step n of generation runs the model over T+n tokens, and for every position already in the
// sequence the hidden state is identical to last step's — same token, same position, same weights,
// therefore the same k and v. A 25-token prompt and 50 generated tokens costs 2,525 position-passes
// where 75 would do (E12.S1.T1).
//
// *** Q is never cached. *** A query is asked once: position 5's query produced position 5's output
// at the step it was generated, and no later step asks it again — later steps ask their own query
// against all keys. Keys and values are read by every future position; queries are not, and that
// asymmetry is the whole design.
//
// Allocated once at the context limit rather than grown: growing per step would reallocate and copy
// the whole history every token, which is the same quadratic in memory traffic.
class KVCache
{
public:
    KVCache(size_t layers, size_t batch, size_t kv_heads, size_t max_positions, size_t head_dim);

    // Writes [B, Hkv, T, Dh] at the current offset. Every layer appends at the same offset, which
    // is why advancing is a separate call: advancing inside append would leave layer 1 one position
    // ahead of layer 0.
    void append(size_t layer, const core::Tensor& keys, const core::Tensor& values);

    // Views over the valid prefix — a slice, no copy (E1.S2.T7). Attention sees [B, Hkv, used, Dh]
    // and neither knows nor cares that the storage is larger.
    core::Tensor keys(size_t layer) const;
    core::Tensor values(size_t layer) const;

    // The prefix of an explicit length. The caller needs this because advancing is separate from
    // appending: within a step the positions just written are not yet counted in used(), and
    // attention must see them — the new token has to attend to itself.
    core::Tensor keys(size_t layer, size_t length) const;
    core::Tensor values(size_t layer, size_t length) const;

    void advance(size_t positions);
    void reset() noexcept { used_ = 0; }

    size_t used() const noexcept { return used_; }
    size_t layers() const noexcept { return keys_.size(); }
    size_t max_positions() const noexcept { return max_positions_; }

private:
    std::vector<core::Tensor> keys_;
    std::vector<core::Tensor> values_;
    size_t max_positions_;
    size_t used_ = 0;
};

} // namespace veda::model

#endif //VEDA_KV_CACHE_H
