#ifndef VEDA_EMBEDDING_H
#define VEDA_EMBEDDING_H

#include "Shape.h"
#include "Tensor.h"

#include <cstdint>
#include <vector>

// Layers: the things that own weights.
//
// veda::ops is pure functions — inputs in, tensor out, no state. veda::nn is what holds learned
// parameters, and the split is what keeps the maths testable on hand-computed numbers with no
// model file in sight (architecture §2).
//
// AD2: no layer allocates its own weights. They arrive as tensor views into the one buffer the
// loader produced, so the 622 MB embedding table exists exactly once no matter how many things
// reference it. Every layer is therefore constructible with synthetic weights — a 4x3 table typed
// by hand is a complete, valid Embedding.
namespace veda::nn
{

// A lookup table: [V, D], one row per token in the vocabulary.
//
// Embedding a token is a row copy, not a computation. The textbook definition — a matmul by a
// one-hot vector — is the same function and 151,935 multiplications by zero more expensive.
class Embedding
{
public:
    // The table is a view; it is neither owned nor copied. Throws if it is not rank 2.
    explicit Embedding(core::Tensor table);

    // [B, T] ids -> [B, T, D]. The id shape is passed explicitly rather than assumed to be [1, T]:
    // B exists in every shape in Veda and is only ever 1 in practice (AD3), and this is where the
    // batch dimension first appears.
    //
    // Throws if the id count disagrees with the shape, or if any id is outside [0, V).
    core::Tensor forward(const std::vector<int64_t>& ids, const core::Shape& id_shape) const;

    size_t vocab_size() const { return table_.shape()[0]; }
    size_t hidden_size() const { return table_.shape()[1]; }
    const core::Tensor& table() const noexcept { return table_; }

private:
    core::Tensor table_;
};

} // namespace veda::nn

#endif //VEDA_EMBEDDING_H
