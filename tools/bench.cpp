// The E13 milestone, as a runnable command:
//
//     veda_bench <model directory> [--tokens N] [--threads T] [--prompt-len P]
//
// Runs a warm-up pass and then a measured generation, printing where the time went and what
// threading bought — and asserting that the outputs are *bitwise* identical either way.
//
// The rule this whole epic exists for: no optimisation without a before-and-after number.

#include "Binding.h"
#include "Generator.h"
#include "KVCache.h"
#include "LMHead.h"
#include "Matmul.h"
#include "ModelConfig.h"
#include "Profile.h"
#include "Safetensors.h"
#include "Sampling.h"
#include "Shape.h"
#include "Transformer.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace
{

struct Timing
{
    std::vector<int32_t> ids;
    double milliseconds = 0.0;
};

Timing run(const veda::model::Transformer& transformer, const veda::model::LMHead& head,
           const std::vector<int32_t>& prompt, size_t tokens, const veda::io::ModelConfig& config)
{
    veda::generate::GreedySampler sampler;
    const veda::generate::Generator generator(transformer, head, sampler);
    veda::model::KVCache cache(config.num_layers, 1, config.num_kv_heads, prompt.size() + tokens,
                               config.head_dim);

    const auto started = std::chrono::steady_clock::now();
    const veda::generate::Result result = generator.generate_cached(prompt, {tokens, {}}, cache);
    const auto finished = std::chrono::steady_clock::now();

    return {result.generated,
            std::chrono::duration<double, std::milli>(finished - started).count()};
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: veda_bench <model directory> [--tokens N] [--threads T] "
                     "[--prompt-len P]\n";
        return 2;
    }

    const std::filesystem::path directory = argv[1];
    size_t tokens = 32;
    size_t threads = 4;
    size_t prompt_length = 8;

    for (int i = 2; i + 1 < argc; i += 2)
    {
        const std::string flag = argv[i];
        const size_t value = static_cast<size_t>(std::stoul(argv[i + 1]));
        if (flag == "--tokens")
        {
            tokens = value;
        }
        else if (flag == "--threads")
        {
            threads = value;
        }
        else if (flag == "--prompt-len")
        {
            prompt_length = value;
        }
    }

    try
    {
        const veda::io::ModelConfig config =
            veda::io::read_model_config((directory / "config.json").string());
        const veda::io::WeightFile weights =
            veda::io::read_safetensors((directory / "model.safetensors").string());
        const veda::model::Transformer transformer = veda::binding::bind(weights, config);
        const veda::model::LMHead head(config.tie_word_embeddings
                                           ? transformer.embedding().table()
                                           : weights["lm_head.weight"]);

        std::vector<int32_t> prompt;
        for (size_t i = 0; i < prompt_length; ++i)
        {
            prompt.push_back(static_cast<int32_t>((i * 7 + 1) % config.vocab_size));
        }

        std::cout << "model: " << config.num_layers << " layers, D=" << config.hidden_size
                  << ", V=" << config.vocab_size << "\nprompt: " << prompt_length
                  << " tokens, generating " << tokens << "\n\n";

        // A warm-up: one run is not a measurement.
        veda::ops::set_thread_count(1);
        (void)run(transformer, head, prompt, 2, config);

        // --- before -------------------------------------------------------------------------
        veda::profile::active_profile().reset();
        veda::profile::set_profiling_enabled(true);
        const Timing single = run(transformer, head, prompt, tokens, config);
        veda::profile::set_profiling_enabled(false);

        std::cout << "where the time goes (1 thread)\n\n"
                  << veda::profile::active_profile().to_string() << "\n";

        // --- after --------------------------------------------------------------------------
        veda::ops::set_thread_count(threads);
        const Timing threaded = run(transformer, head, prompt, tokens, config);
        veda::ops::set_thread_count(1);

        const bool identical = single.ids == threaded.ids;

        std::cout << "threads " << 1 << "     " << tokens << " tokens in " << single.milliseconds
                  << " ms     " << (1000.0 * static_cast<double>(tokens) / single.milliseconds)
                  << " tok/s\n"
                  << "threads " << threads << "     " << tokens << " tokens in "
                  << threaded.milliseconds << " ms     "
                  << (1000.0 * static_cast<double>(tokens) / threaded.milliseconds) << " tok/s     "
                  << (single.milliseconds / threaded.milliseconds) << "x\n\n"
                  << "outputs identical: " << (identical ? "true" : "FALSE")
                  << "      (bitwise, not within a tolerance)\n";

        if (!identical)
        {
            std::cerr << "\nthe optimisation changed the output — that is a bug, not a speedup\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
