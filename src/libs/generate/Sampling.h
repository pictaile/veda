#ifndef VEDA_SAMPLING_H
#define VEDA_SAMPLING_H

#include "Tensor.h"

#include <cstdint>
#include <random>
#include <vector>

// Turning logits into a token is a separate concern from producing them: the choice depends on what
// the text is for, not on what the model is.
//
// A sampler sees only logits — not the model, not the prompt, not the tokenizer. That narrowness is
// what the architecture's §7 names as one of three things Phase 1 must not foreclose: a
// constrained-decoding hook ("only emit tokens that keep this JSON valid") is a predicate over the
// logits tensor rather than a redesign, precisely because the sampler is already a separate object
// that sees only logits.
// The header is Sampling.h rather than Sampler.h: the project scaffold already has a Sampler.h in
// src/libs/sampler/, and two headers with one name in the include path is a coin toss.
namespace veda::generate
{

class Sampler
{
public:
    virtual ~Sampler() = default;

    // [V] or [1, V] -> a token id.
    //
    // Non-const on purpose: greedy has no state, but temperature and top-p carry a random engine,
    // and an interface whose implementations need a mutable hack is worse than one that admits
    // sampling advances state.
    virtual int32_t sample(const core::Tensor& logits) = 0;
};

// argmax, always — and argmax needs no softmax, because softmax is monotonic. At V = 151,936 that
// saves 151,936 exponentials per token for the same answer, which is why the LM head deliberately
// does not normalise (E10.S3.T4).
//
// Ties go to the lower id. Mathematically unimportant, and important for reproducibility: without a
// stated rule two implementations can disagree on a tie and produce different text from the same
// model.
class GreedySampler final : public Sampler
{
public:
    int32_t sample(const core::Tensor& logits) override;
};

// Temperature is a division before the exponential:  softmax(logits / T).
//
// The order of the logits never changes — dividing by a positive number is monotonic — so
// temperature changes the *gaps*, and softmax exponentiates gaps. T -> 0 is greedy, T -> infinity
// is uniform, which is to say "ignore the model". It is not creativity; it is how much of the
// model's own uncertainty to honour.
//
// T = 0 is accepted and short-circuits to argmax rather than dividing by zero.
class TemperatureSampler final : public Sampler
{
public:
    TemperatureSampler(float temperature, uint64_t seed);
    int32_t sample(const core::Tensor& logits) override;

private:
    float temperature_;
    std::mt19937_64 engine_;
};

// Keep the k most likely, renormalise, sample.
//
// Simple, and blind to the shape of the distribution: the same k is too loose when the model is
// certain (39 wrong answers sharing 3% of the mass) and too tight when it is not (10 plausible
// continuations discarded). k is a count, and confidence is not.
class TopKSampler final : public Sampler
{
public:
    TopKSampler(size_t k, float temperature, uint64_t seed);
    int32_t sample(const core::Tensor& logits) override;

private:
    size_t k_;
    float temperature_;
    std::mt19937_64 engine_;
};

// Nucleus sampling: keep the smallest set whose probabilities sum to at least p.
//
// The count adapts — one token when the model is sure, hundreds when it is not — which is the
// behaviour top-k was approximating with a constant. The token that *crosses* the threshold is
// included, since the set must reach p; and the nucleus is never empty, so a confident distribution
// with a small p still leaves something to sample.
class TopPSampler final : public Sampler
{
public:
    TopPSampler(float p, float temperature, uint64_t seed);
    int32_t sample(const core::Tensor& logits) override;

private:
    float p_;
    float temperature_;
    std::mt19937_64 engine_;
};

} // namespace veda::generate

#endif //VEDA_SAMPLING_H
