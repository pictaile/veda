// The E4 milestone, as a runnable command:
//
//     veda_inspect <model directory>
//     veda_inspect <model directory>/model.safetensors
//
// Loads config.json and the weight file and prints what was read — the dimensions, then every
// tensor's name and shape. It is the first sanity check a person performs against a real model, and
// it exercises the whole of veda::io without a single line from nn or model.

#include "ModelConfig.h"
#include "Safetensors.h"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

namespace
{

std::string human_bytes(size_t bytes)
{
    const char* units[] = {"B", "KB", "MB", "GB"};
    double value = static_cast<double>(bytes);
    int unit = 0;
    while (value >= 1024.0 && unit < 3)
    {
        value /= 1024.0;
        ++unit;
    }
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.1f %s", value, units[unit]);
    return buffer;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "usage: veda_inspect <model directory or .safetensors file>\n";
        return 2;
    }

    const std::filesystem::path target = argv[1];
    const bool is_directory = std::filesystem::is_directory(target);
    const std::filesystem::path weights_path =
        is_directory ? target / "model.safetensors" : target;
    const std::filesystem::path config_path =
        is_directory ? target / "config.json" : target.parent_path() / "config.json";

    try
    {
        if (std::filesystem::exists(config_path))
        {
            std::cout << veda::io::read_model_config(config_path.string()).to_string() << "\n\n";
        }
        else
        {
            std::cout << "no config.json beside the weights; printing tensors only\n\n";
        }

        const veda::io::WeightFile weights = veda::io::read_safetensors(weights_path.string());

        for (const std::string& name : weights.names())
        {
            std::cout << "  " << std::left << std::setw(52) << name
                      << weights[name].shape().to_string() << "\n";
        }

        std::cout << "\n" << weights.tensor_count() << " tensors, " << human_bytes(weights.bytes_on_disk())
                  << " on disk, " << human_bytes(weights.bytes_resident()) << " resident as fp32\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
