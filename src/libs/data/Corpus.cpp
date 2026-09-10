#include "Corpus.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace veda::data
{

using core::Shape;

namespace
{
size_t split_begin(size_t boundary, size_t total, Split split)
{
    return split == Split::Train ? 0 : boundary;
}

size_t split_end(size_t boundary, size_t total, Split split)
{
    return split == Split::Train ? boundary : total;
}
} // namespace

Corpus::Corpus(std::vector<int64_t> ids, double validation_fraction, uint64_t seed)
    : ids_(std::move(ids)), state_(seed == 0 ? 0x9E3779B97F4A7C15ull : seed)
{
    if (validation_fraction < 0.0 || validation_fraction > 1.0)
    {
        throw std::invalid_argument("Corpus: validation_fraction must lie in [0, 1]");
    }

    // Rounded once, here, and stored. Recomputing it per batch invites the boundary to move.
    const double kept = 1.0 - validation_fraction;
    boundary_ = static_cast<size_t>(static_cast<double>(ids_.size()) * kept);
    if (boundary_ > ids_.size())
    {
        boundary_ = ids_.size();
    }
}

Corpus Corpus::from_text(std::string_view text, const tokenizer::Tokenizer& tokenizer,
                         double validation_fraction, uint64_t seed)
{
    const std::vector<int32_t> encoded = tokenizer.encode(text);
    std::vector<int64_t> ids(encoded.begin(), encoded.end());
    return Corpus(std::move(ids), validation_fraction, seed);
}

size_t Corpus::size(Split split) const noexcept
{
    return split_end(boundary_, ids_.size(), split) - split_begin(boundary_, ids_.size(), split);
}

uint64_t Corpus::next_random()
{
    // splitmix64: small, well-distributed, and reproducible across platforms — the same reasons
    // E11 gave for not using the standard library's engines in a place that must be exactly
    // repeatable.
    state_ += 0x9E3779B97F4A7C15ull;
    uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

Batch Corpus::next_batch(size_t batch_size, size_t length, Split split)
{
    if (batch_size == 0 || length == 0)
    {
        throw std::invalid_argument("Corpus: batch size and length must be positive");
    }

    const size_t begin = split_begin(boundary_, ids_.size(), split);
    const size_t end = split_end(boundary_, ids_.size(), split);
    const size_t available = end - begin;

    // T inputs and their T successors: T + 1 ids per window.
    const size_t window = length + 1;
    if (available < window)
    {
        throw std::invalid_argument(
            "Corpus: the " + std::string(split == Split::Train ? "training" : "validation") +
            " split holds " + std::to_string(available) + " ids, and a window of length " +
            std::to_string(length) + " needs " + std::to_string(window));
    }

    const size_t offsets = available - window + 1;   // how many legal starting positions there are

    Batch batch;
    batch.shape = Shape({batch_size, length});
    batch.inputs.resize(batch_size * length);
    batch.targets.resize(batch_size * length);

    for (size_t row = 0; row < batch_size; ++row)
    {
        // Drawn per row, so one batch mixes windows from all over the split.
        const size_t start = begin + static_cast<size_t>(next_random() % offsets);
        for (size_t t = 0; t < length; ++t)
        {
            batch.inputs[row * length + t] = ids_[start + t];
            batch.targets[row * length + t] = ids_[start + t + 1];
        }
    }

    return batch;
}

} // namespace veda::data
