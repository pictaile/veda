#ifndef VEDA_CORPUS_H
#define VEDA_CORPUS_H

#include "Shape.h"
#include "Tokenizer.h"

#include <cstdint>
#include <string_view>
#include <vector>

// Text in, [B, T] batches out.
//
// THE TEXT LABELS ITSELF. This is the idea the whole of language modelling rests on: the target for
// position t is simply the token at t+1, so any text at all is training data — no annotation, no
// labels, no dataset construction.
//
//     text     T  h  e  ' '  c  a  t
//     inputs   T  h  e  ' '  c  a
//     targets  h  e  ' '  c  a  t
//
// And because of the causal mask (E7), ONE forward pass over a window of length T produces T
// training examples at once: position 0 is scored on predicting position 1 while position 5 is
// scored on predicting position 6, and neither can see its own answer. That is what the mask was
// for. A window of length T therefore needs T + 1 ids.
namespace veda::data
{

enum class Split
{
    Train,
    Validation
};

struct Batch
{
    std::vector<int64_t> inputs;    // B*T ids
    std::vector<int64_t> targets;   // B*T ids, each the successor of the input beside it
    core::Shape shape;              // [B, T]
};

class Corpus
{
public:
    // The split is BY POSITION, not by window. Splitting randomly — some windows to train, some to
    // validate — leaks: two windows overlapping by one token share their answer, and the validation
    // loss becomes a measurement of memorisation rather than of generalisation. The corpus is cut
    // once, into two contiguous regions, and windows are drawn inside each.
    //
    // The consequence is worth stating rather than hiding: with a small corpus the two regions come
    // from different PARTS of the text, so a gap between train and validation loss may be a
    // difference in subject matter rather than in the model. That is a limitation of small-corpus
    // training, not a bug.
    Corpus(std::vector<int64_t> ids, double validation_fraction, uint64_t seed);

    static Corpus from_text(std::string_view text, const tokenizer::Tokenizer& tokenizer,
                            double validation_fraction, uint64_t seed);

    // Windows at RANDOM OFFSETS, not consecutive blocks. Partitioning the corpus into blocks of T
    // shows the model one alignment of the text to window boundaries, and a phrase that straddles a
    // boundary is never seen whole. Random offsets show every alignment, at no cost.
    //
    // Throws std::invalid_argument if the split cannot hold a single window of T + 1 ids — better
    // than returning a batch that silently repeats the only window that fits.
    Batch next_batch(size_t batch_size, size_t length, Split split);

    size_t size(Split split) const noexcept;
    const std::vector<int64_t>& ids() const noexcept { return ids_; }

    // Where the training region ends and validation begins, as an index into ids().
    size_t boundary() const noexcept { return boundary_; }

private:
    std::vector<int64_t> ids_;
    size_t boundary_ = 0;

    // Its own generator, seeded — like E11's sampler, and for the same reason: a training run that
    // cannot be repeated cannot be debugged.
    uint64_t state_ = 0;

    uint64_t next_random();
};

} // namespace veda::data

#endif //VEDA_CORPUS_H
