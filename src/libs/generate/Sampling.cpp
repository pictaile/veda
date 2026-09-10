#include "Sampling.h"

#include "Shape.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace veda::generate
{

using core::Tensor;

namespace
{

// LMHead::forward_last returns [B, V] and the loop passes one row of it; accepting both shapes
// saves a reshape at every call site.
size_t vocabulary_size(const Tensor& logits)
{
    if (logits.rank() == 1)
    {
        return logits.shape()[0];
    }
    if (logits.rank() == 2 && logits.shape()[0] == 1)
    {
        return logits.shape()[1];
    }
    throw std::invalid_argument("generate: logits must be [V] or [1, V], got " +
                                logits.shape().to_string());
}

float logit_at(const Tensor& logits, size_t index)
{
    return logits.rank() == 1 ? logits.at({index}) : logits.at({0, index});
}

// The shared pipeline, written once: three copies of softmax-with-temperature would be three
// places for the temperature to be applied twice.
//
// Scale, then softmax — and never the other way round, nor truncate before scaling: scaling changes
// the probabilities a threshold is applied to, so truncating first would select a different set.
std::vector<float> distribution_of(const Tensor& logits, size_t vocabulary, float temperature)
{
    std::vector<float> probabilities(vocabulary);

    float maximum = -std::numeric_limits<float>::infinity();
    for (size_t v = 0; v < vocabulary; ++v)
    {
        const float value = logit_at(logits, v);
        if (std::isnan(value))
        {
            throw std::runtime_error("generate: logit " + std::to_string(v) + " is NaN");
        }
        probabilities[v] = value / temperature;
        maximum = std::max(maximum, probabilities[v]);
    }

    // The same stability argument as E2.S3.T8: exp overflows fp32 just past 88, and scaled logits
    // reach that easily at a low temperature.
    float total = 0.0f;
    for (float& value : probabilities)
    {
        value = std::exp(value - maximum);
        total += value;
    }
    for (float& value : probabilities)
    {
        value /= total;
    }
    return probabilities;
}

// Walks the cumulative sum against a uniform draw. The probabilities must already sum to 1 —
// renormalising after truncation is what makes this correct, and skipping it biases every draw
// towards the first token in a way that is invisible at k = 1.
int32_t draw_from(const std::vector<float>& probabilities, const std::vector<size_t>& ids,
                  std::mt19937_64& engine)
{
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    const float target = uniform(engine);

    float cumulative = 0.0f;
    for (size_t i = 0; i < ids.size(); ++i)
    {
        cumulative += probabilities[i];
        if (target < cumulative)
        {
            return static_cast<int32_t>(ids[i]);
        }
    }
    return static_cast<int32_t>(ids.back());   // floating-point slack at the very top
}

int32_t greedy_of(const Tensor& logits, size_t vocabulary)
{
    size_t best = 0;
    for (size_t v = 0; v < vocabulary; ++v)
    {
        const float value = logit_at(logits, v);
        if (std::isnan(value))
        {
            throw std::runtime_error("generate: logit " + std::to_string(v) + " is NaN");
        }
        if (v > 0 && value > logit_at(logits, best))
        {
            best = v;
        }
    }
    return static_cast<int32_t>(best);
}

// The kept ids and their renormalised probabilities, in descending order of probability.
void truncate(std::vector<float>& probabilities, std::vector<size_t>& ids, size_t keep)
{
    std::vector<size_t> order(probabilities.size());
    for (size_t i = 0; i < order.size(); ++i)
    {
        order[i] = i;
    }
    keep = std::min(keep, order.size());
    std::partial_sort(order.begin(), order.begin() + static_cast<long>(keep), order.end(),
                      [&](size_t a, size_t b) {
                          return probabilities[a] != probabilities[b] ? probabilities[a] > probabilities[b]
                                                                      : a < b;
                      });

    std::vector<float> kept;
    kept.reserve(keep);
    ids.clear();
    ids.reserve(keep);

    float total = 0.0f;
    for (size_t i = 0; i < keep; ++i)
    {
        kept.push_back(probabilities[order[i]]);
        ids.push_back(order[i]);
        total += kept.back();
    }
    for (float& value : kept)
    {
        value /= total;
    }
    probabilities = std::move(kept);
}

} // namespace

int32_t GreedySampler::sample(const Tensor& logits)
{
    const size_t vocabulary = vocabulary_size(logits);
    if (vocabulary == 0)
    {
        throw std::invalid_argument("generate: cannot sample from an empty vocabulary");
    }
    // A NaN is a bug upstream, not something to sample around: every comparison with NaN is false,
    // so a naive argmax would silently return index 0. Ties go to the lower id.
    return greedy_of(logits, vocabulary);
}

TemperatureSampler::TemperatureSampler(float temperature, uint64_t seed)
    : temperature_(temperature), engine_(seed)
{
    if (!(temperature_ >= 0.0f))
    {
        throw std::invalid_argument("generate: temperature must be >= 0");
    }
}

int32_t TemperatureSampler::sample(const Tensor& logits)
{
    const size_t vocabulary = vocabulary_size(logits);
    if (vocabulary == 0)
    {
        throw std::invalid_argument("generate: cannot sample from an empty vocabulary");
    }
    if (temperature_ == 0.0f)
    {
        return greedy_of(logits, vocabulary);
    }

    std::vector<float> probabilities = distribution_of(logits, vocabulary, temperature_);
    std::vector<size_t> ids(vocabulary);
    for (size_t v = 0; v < vocabulary; ++v)
    {
        ids[v] = v;
    }
    return draw_from(probabilities, ids, engine_);
}

TopKSampler::TopKSampler(size_t k, float temperature, uint64_t seed)
    : k_(k), temperature_(temperature), engine_(seed)
{
    if (k_ == 0)
    {
        throw std::invalid_argument("generate: k must be positive");
    }
    if (!(temperature_ >= 0.0f))
    {
        throw std::invalid_argument("generate: temperature must be >= 0");
    }
}

int32_t TopKSampler::sample(const Tensor& logits)
{
    const size_t vocabulary = vocabulary_size(logits);
    if (vocabulary == 0)
    {
        throw std::invalid_argument("generate: cannot sample from an empty vocabulary");
    }
    if (temperature_ == 0.0f || k_ == 1)
    {
        return greedy_of(logits, vocabulary);
    }

    std::vector<float> probabilities = distribution_of(logits, vocabulary, temperature_);
    std::vector<size_t> ids;
    truncate(probabilities, ids, k_);
    return draw_from(probabilities, ids, engine_);
}

TopPSampler::TopPSampler(float p, float temperature, uint64_t seed)
    : p_(p), temperature_(temperature), engine_(seed)
{
    if (!(p_ >= 0.0f) || p_ > 1.0f)
    {
        throw std::invalid_argument("generate: p must be in [0, 1]");
    }
    if (!(temperature_ >= 0.0f))
    {
        throw std::invalid_argument("generate: temperature must be >= 0");
    }
}

int32_t TopPSampler::sample(const Tensor& logits)
{
    const size_t vocabulary = vocabulary_size(logits);
    if (vocabulary == 0)
    {
        throw std::invalid_argument("generate: cannot sample from an empty vocabulary");
    }
    if (temperature_ == 0.0f)
    {
        return greedy_of(logits, vocabulary);
    }

    std::vector<float> probabilities = distribution_of(logits, vocabulary, temperature_);

    // Sort everything, then take the prefix that reaches p. The token that crosses the threshold is
    // included — the set must sum to at least p — and the nucleus is never empty, so a confident
    // distribution with a small p still leaves something to sample.
    std::vector<size_t> ids;
    truncate(probabilities, ids, vocabulary);

    size_t nucleus = 1;
    float cumulative = probabilities[0];
    while (nucleus < probabilities.size() && cumulative < p_)
    {
        cumulative += probabilities[nucleus];
        ++nucleus;
    }

    probabilities.resize(nucleus);
    ids.resize(nucleus);

    float total = 0.0f;
    for (const float value : probabilities)
    {
        total += value;
    }
    for (float& value : probabilities)
    {
        value /= total;
    }

    return draw_from(probabilities, ids, engine_);
}

} // namespace veda::generate
