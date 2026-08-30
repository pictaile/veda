#ifndef VEDA_SAFETENSORS_H
#define VEDA_SAFETENSORS_H

#include "Tensor.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

// Reading model weights, knowing nothing about transformers.
//
// The file is three parts: an 8-byte little-endian header length, that many bytes of JSON naming
// every tensor's dtype, shape and byte range, and then the raw data. Nothing in it is executable —
// which is the reason the format exists, and the reason a model can be downloaded without running
// a stranger's code.
//
// Weights arrive as bf16 and are widened to fp32 on load (AD1), so no op above this layer ever
// branches on a dtype. This module produces a name-to-tensor map and stops: mapping those names
// onto model layers is `binding`, and it cannot exist before the layers do (E10).
namespace veda::io
{

class WeightFile
{
public:
    // Throws std::runtime_error naming the tensor if it is not in the file.
    const core::Tensor& operator[](const std::string& name) const;
    bool contains(const std::string& name) const;

    // In header order, which is how the file was written — occasionally the fastest way to notice
    // that two files came from different exports.
    const std::vector<std::string>& names() const noexcept { return names_; }

    size_t tensor_count() const noexcept { return entries_.size(); }
    size_t bytes_on_disk() const noexcept { return bytes_on_disk_; }
    size_t bytes_resident() const noexcept { return bytes_resident_; }

    // Used by the reader.
    void add(std::string name, core::Tensor tensor, size_t on_disk);

private:
    std::vector<std::pair<std::string, core::Tensor>> entries_;
    std::vector<std::string> names_;
    size_t bytes_on_disk_ = 0;
    size_t bytes_resident_ = 0;
};

// Reads every tensor in the file. Throws std::runtime_error, naming the path and the reason, for a
// file too small to hold a header, a header length beyond the file, a header that is not JSON, an
// unsupported dtype, a byte range that disagrees with the shape, or one that runs past the data
// section. An unknown dtype fails loudly and is never reinterpreted.
WeightFile read_safetensors(const std::string& path);

} // namespace veda::io

#endif //VEDA_SAFETENSORS_H
