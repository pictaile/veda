#ifndef VEDA_GENERATOR_H
#define VEDA_GENERATOR_H

#include "KVCache.h"
#include "LMHead.h"
#include "Sampling.h"
#include "Transformer.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace veda::generate
{

struct Options
{
    size_t max_new_tokens = 64;
    std::vector<int32_t> stop_tokens;   // usually the EOS id
};

struct Result
{
    std::vector<int32_t> generated;
    bool stopped_on_token = false;   // as opposed to hitting the limit
    size_t forward_passes = 0;
};

// Autoregression: the output becomes the input.
//
//     logits = head(transformer(ids))      the last position only
//     next   = sampler(logits)
//     ids.push_back(next)
//
// Each step conditions on everything said so far, including what the model itself just said — which
// is why an error at step 3 shapes steps 4 through 50. The model is not wrong about the prompt; it
// is answering a prompt that now contains its own mistake.
//
// *** Without a KV cache every step reprocesses the whole prefix. *** Step n runs the transformer
// over all T+n tokens although T+n-1 of them were processed identically the step before: every
// layer recomputes every key and value for every earlier position, and attention then discards all
// but the last row of scores. A 25-token prompt plus 50 generated tokens costs 2,525
// position-passes where a cache would cost 75. That is the roadmap's risk R3, it is expected rather
// than a bug, and it is what E12 removes.
//
// The generator knows the model and the sampler and nothing else — it does not tokenize. The caller
// brings ids and turns them back into text (StreamingDecoder, E5.S3.T5), which keeps this loop
// testable with a synthetic model and no tokenizer at all.
class Generator
{
public:
    Generator(const model::Transformer& transformer, const model::LMHead& head, Sampler& sampler);

    // The callback fires per emitted token, in order — at roughly a second per token that is the
    // difference between a program that looks hung and one that looks slow. The stop token is never
    // emitted: it marks the end, and including it would put <|endoftext|> in the user's text.
    Result generate(const std::vector<int32_t>& prompt, const Options& options,
                    const std::function<void(int32_t)>& on_token = {}) const;

    // The same loop with a KV cache: the prompt is prefilled once, and every step feeds a single
    // token. Token for token identical to generate() — that identity is what proves the cache
    // correct, and it is why the uncached path stays (roadmap acceptance criterion).
    Result generate_cached(const std::vector<int32_t>& prompt, const Options& options,
                           model::KVCache& cache,
                           const std::function<void(int32_t)>& on_token = {}) const;

private:
    const model::Transformer& transformer_;
    const model::LMHead& head_;
    Sampler& sampler_;
};

} // namespace veda::generate

#endif //VEDA_GENERATOR_H
