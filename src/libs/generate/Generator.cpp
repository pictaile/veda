#include "Generator.h"

#include "Shape.h"

#include <algorithm>
#include <stdexcept>

namespace veda::generate
{

using core::Shape;
using core::Tensor;

Generator::Generator(const model::Transformer& transformer, const model::LMHead& head,
                     Sampler& sampler)
    : transformer_(transformer), head_(head), sampler_(sampler)
{
}

Result Generator::generate(const std::vector<int32_t>& prompt, const Options& options,
                           const std::function<void(int32_t)>& on_token) const
{
    if (prompt.empty())
    {
        throw std::invalid_argument("generate: the prompt is empty — there is nothing to condition on");
    }

    Result result;

    // The whole sequence so far. Every step re-runs the model over all of it (E12 fixes that).
    std::vector<int64_t> ids(prompt.begin(), prompt.end());

    for (size_t step = 0; step < options.max_new_tokens; ++step)
    {
        const Tensor hidden = transformer_.forward(ids, Shape({1, ids.size()}));
        const Tensor logits = head_.forward_last(hidden);
        ++result.forward_passes;

        const int32_t next = sampler_.sample(logits);

        if (std::find(options.stop_tokens.begin(), options.stop_tokens.end(), next) !=
            options.stop_tokens.end())
        {
            result.stopped_on_token = true;
            break;
        }

        result.generated.push_back(next);
        ids.push_back(next);

        if (on_token)
        {
            on_token(next);
        }
    }

    return result;
}

Result Generator::generate_cached(const std::vector<int32_t>& prompt, const Options& options,
                                  model::KVCache& cache,
                                  const std::function<void(int32_t)>& on_token) const
{
    if (prompt.empty())
    {
        throw std::invalid_argument("generate: the prompt is empty — there is nothing to condition on");
    }

    Result result;
    cache.reset();

    // Prefill: the prompt in one pass. Unavoidable — the cache removes the *re*-processing, not the
    // processing.
    std::vector<int64_t> pending(prompt.begin(), prompt.end());

    for (size_t step = 0; step <= options.max_new_tokens; ++step)
    {
        const Tensor hidden =
            transformer_.forward_cached(pending, Shape({1, pending.size()}), cache);
        const Tensor logits = head_.forward_last(hidden);
        ++result.forward_passes;

        if (step == options.max_new_tokens)
        {
            break;   // the prefill still had to run; this is the limit, not an extra token
        }

        const int32_t next = sampler_.sample(logits);

        if (std::find(options.stop_tokens.begin(), options.stop_tokens.end(), next) !=
            options.stop_tokens.end())
        {
            result.stopped_on_token = true;
            break;
        }

        result.generated.push_back(next);
        if (on_token)
        {
            on_token(next);
        }

        pending = {next};   // every step after the prefill feeds exactly one token
    }

    return result;
}

} // namespace veda::generate
