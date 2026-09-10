// The E11 milestone, as a runnable command:
//
//     veda_run <model directory> [options] --prompt "text"
//     veda_run <model directory> [options] --ids 3 1 4 1 5
//
//     --max N        how many tokens to generate      (default 32)
//     --temp T       temperature, 0 means greedy      (default 0)
//     --top-p P      nucleus sampling                 (default off)
//     --top-k K      keep the k most likely           (default off)
//     --seed S       for reproducibility              (default 0)
//
// Loads config.json, model.safetensors and (for --prompt) tokenizer.json, binds every weight to a
// layer, and generates. Output is streamed through the decoder that holds back partial UTF-8
// characters, so nothing invalid ever reaches the terminal.

#include "Binding.h"
#include "Generator.h"
#include "KVCache.h"
#include "LMHead.h"
#include "ModelConfig.h"
#include "Safetensors.h"
#include "Sampling.h"
#include "Shape.h"
#include "Tensor.h"
#include "Tokenizer.h"
#include "Transformer.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace
{

struct Arguments
{
    std::filesystem::path directory;
    std::string prompt;
    std::vector<int32_t> ids;
    size_t max_new_tokens = 32;
    float temperature = 0.0f;
    float top_p = 0.0f;
    size_t top_k = 0;
    uint64_t seed = 0;
    bool no_cache = false;
};

std::unique_ptr<veda::generate::Sampler> make_sampler(const Arguments& arguments)
{
    if (arguments.temperature == 0.0f)
    {
        return std::make_unique<veda::generate::GreedySampler>();
    }
    if (arguments.top_k > 0)
    {
        return std::make_unique<veda::generate::TopKSampler>(arguments.top_k, arguments.temperature,
                                                             arguments.seed);
    }
    if (arguments.top_p > 0.0f)
    {
        return std::make_unique<veda::generate::TopPSampler>(arguments.top_p, arguments.temperature,
                                                             arguments.seed);
    }
    return std::make_unique<veda::generate::TemperatureSampler>(arguments.temperature,
                                                                arguments.seed);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: veda_run <model directory> [--max N] [--temp T] [--top-p P] "
                     "[--top-k K] [--seed S] (--prompt \"text\" | --ids 1 2 3)\n";
        return 2;
    }

    Arguments arguments;
    arguments.directory = argv[1];

    for (int i = 2; i < argc; ++i)
    {
        const std::string flag = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };

        if (flag == "--prompt")
        {
            arguments.prompt = next();
        }
        else if (flag == "--ids")
        {
            while (i + 1 < argc && argv[i + 1][0] != '-')
            {
                arguments.ids.push_back(static_cast<int32_t>(std::stoi(argv[++i])));
            }
        }
        else if (flag == "--max")
        {
            arguments.max_new_tokens = static_cast<size_t>(std::stoul(next()));
        }
        else if (flag == "--temp")
        {
            arguments.temperature = std::stof(next());
        }
        else if (flag == "--top-p")
        {
            arguments.top_p = std::stof(next());
        }
        else if (flag == "--top-k")
        {
            arguments.top_k = static_cast<size_t>(std::stoul(next()));
        }
        else if (flag == "--seed")
        {
            arguments.seed = std::stoull(next());
        }
        else if (flag == "--no-cache")
        {
            // The uncached path is kept as the reference the cache is judged against (E12).
            arguments.no_cache = true;
        }
        else
        {
            // A bare number after the directory is a token id, so the E10 form still works.
            arguments.ids.push_back(static_cast<int32_t>(std::stoi(flag)));
        }
    }

    try
    {
        const auto started = std::chrono::steady_clock::now();

        const veda::io::ModelConfig config =
            veda::io::read_model_config((arguments.directory / "config.json").string());
        const veda::io::WeightFile weights =
            veda::io::read_safetensors((arguments.directory / "model.safetensors").string());
        const veda::model::Transformer transformer = veda::binding::bind(weights, config);
        const veda::model::LMHead head(config.tie_word_embeddings
                                           ? transformer.embedding().table()
                                           : weights["lm_head.weight"]);

        std::cout << config.to_string() << "\n\nloaded " << weights.tensor_count() << " tensors, "
                  << weights.bytes_resident() / (1024 * 1024) << " MB resident\n";

        // The tokenizer is only needed for text; --ids works without one.
        std::unique_ptr<veda::tokenizer::Tokenizer> tokenizer;
        const std::filesystem::path tokenizer_path = arguments.directory / "tokenizer.json";
        if (std::filesystem::exists(tokenizer_path))
        {
            std::ifstream in(tokenizer_path);
            std::ostringstream buffer;
            buffer << in.rdbuf();
            tokenizer = std::make_unique<veda::tokenizer::Tokenizer>(
                veda::tokenizer::Tokenizer::from_tokenizer_json(
                    veda::io::parse_json(buffer.str())));
            std::cout << "tokenizer: " << tokenizer->vocabulary().size() << " tokens, "
                      << tokenizer->merges().size() << " merges\n";
        }

        std::vector<int32_t> prompt = arguments.ids;
        if (!arguments.prompt.empty())
        {
            if (!tokenizer)
            {
                throw std::runtime_error("--prompt needs tokenizer.json beside the weights");
            }
            prompt = tokenizer->encode(arguments.prompt);
        }
        if (prompt.empty())
        {
            throw std::runtime_error("nothing to run: pass --prompt or --ids");
        }

        const auto loaded = std::chrono::steady_clock::now();
        std::cout << "\nprompt: " << prompt.size() << " tokens\n\n";
        if (!arguments.prompt.empty())
        {
            std::cout << arguments.prompt;
        }
        std::cout.flush();

        std::unique_ptr<veda::generate::Sampler> sampler = make_sampler(arguments);
        const veda::generate::Generator generator(transformer, head, *sampler);

        // Held-back partial characters never reach the terminal (E5.S3.T5).
        veda::tokenizer::StreamingDecoder decoder(
            tokenizer ? *tokenizer
                      : *(tokenizer = std::make_unique<veda::tokenizer::Tokenizer>(
                              veda::tokenizer::Vocabulary{}, veda::tokenizer::Merges{})));

        const auto emit = [&](int32_t id) {
            if (!arguments.prompt.empty())
            {
                std::cout << decoder.push(id) << std::flush;
            }
            else
            {
                std::cout << id << " " << std::flush;
            }
        };

        const veda::generate::Options options{arguments.max_new_tokens, {}};

        veda::generate::Result result;
        if (arguments.no_cache)
        {
            result = generator.generate(prompt, options, emit);
        }
        else
        {
            // Allocated once at the context limit, or at what this run could possibly need.
            const size_t limit = config.max_position_embeddings > 0
                                     ? config.max_position_embeddings
                                     : prompt.size() + arguments.max_new_tokens;
            veda::model::KVCache cache(config.num_layers, 1, config.num_kv_heads,
                                       std::min(limit, prompt.size() + arguments.max_new_tokens),
                                       config.head_dim);
            result = generator.generate_cached(prompt, options, cache, emit);
        }

        if (!arguments.prompt.empty())
        {
            std::cout << decoder.flush();
        }

        const auto done = std::chrono::steady_clock::now();
        const auto milliseconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(done - loaded).count();

        std::cout << "\n\n" << result.generated.size() << " tokens in " << milliseconds << " ms";
        if (milliseconds > 0)
        {
            std::cout << "  (" << (1000.0 * static_cast<double>(result.generated.size()) /
                                   static_cast<double>(milliseconds))
                      << " tok/s)";
        }
        std::cout << "\n" << result.forward_passes << " forward passes"
                  << (result.stopped_on_token ? ", stopped on a stop token" : ", hit the limit")
                  << "\nload and bind: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(loaded - started).count()
                  << " ms\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
