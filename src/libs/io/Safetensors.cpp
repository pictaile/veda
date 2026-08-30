#include "Safetensors.h"

#include "BFloat16.h"
#include "DType.h"
#include "Json.h"
#include "Shape.h"

#include <bit>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace veda::io
{

using core::DType;
using core::Shape;
using core::Tensor;

namespace
{

[[noreturn]] void fail(const std::string& path, const std::string& reason)
{
    throw std::runtime_error("safetensors \"" + path + "\": " + reason);
}

void read_exact(std::istream& in, char* destination, size_t count, const std::string& path,
                const std::string& what)
{
    if (!in.read(destination, static_cast<std::streamsize>(count)))
    {
        fail(path, "truncated while reading " + what);
    }
}

// Little-endian, assembled byte by byte. Every platform Veda targets is little-endian anyway; being
// explicit here costs nothing and makes the format self-describing in the code.
uint64_t little_endian_u64(const unsigned char* bytes)
{
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i)
    {
        value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    }
    return value;
}

Shape shape_from(const JsonValue& dims, const std::string& path, const std::string& name)
{
    if (!dims.is_array())
    {
        fail(path, "tensor \"" + name + "\": shape is not an array");
    }
    std::vector<size_t> extents;
    extents.reserve(dims.size());
    for (size_t k = 0; k < dims.size(); ++k)
    {
        const int64_t extent = dims[k].as_int();
        if (extent < 0)
        {
            fail(path, "tensor \"" + name + "\": negative dimension");
        }
        extents.push_back(static_cast<size_t>(extent));
    }
    // Shape itself refuses a dimension product that would wrap around (E1.S3.T9), which is what a
    // corrupt header would otherwise turn into a silent under-allocation.
    return Shape(std::move(extents));
}

} // namespace

void WeightFile::add(std::string name, Tensor tensor, size_t on_disk)
{
    bytes_on_disk_ += on_disk;
    bytes_resident_ += tensor.numel() * sizeof(float);
    names_.push_back(name);
    entries_.emplace_back(std::move(name), std::move(tensor));
}

const Tensor& WeightFile::operator[](const std::string& name) const
{
    for (const auto& entry : entries_)
    {
        if (entry.first == name)
        {
            return entry.second;
        }
    }
    throw std::runtime_error("safetensors: no tensor named \"" + name + "\"");
}

bool WeightFile::contains(const std::string& name) const
{
    for (const auto& entry : entries_)
    {
        if (entry.first == name)
        {
            return true;
        }
    }
    return false;
}

WeightFile read_safetensors(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        fail(path, "cannot be opened");
    }

    in.seekg(0, std::ios::end);
    const auto file_size = static_cast<uint64_t>(in.tellg());
    in.seekg(0);

    if (file_size < 8)
    {
        fail(path, "is " + std::to_string(file_size) + " bytes, too small to hold a header length");
    }

    unsigned char length_bytes[8];
    read_exact(in, reinterpret_cast<char*>(length_bytes), 8, path, "the header length");
    const uint64_t header_length = little_endian_u64(length_bytes);

    if (header_length > file_size - 8)
    {
        fail(path, "header claims " + std::to_string(header_length) + " bytes, the file has " +
                       std::to_string(file_size - 8) + " after the length prefix");
    }

    std::string header(static_cast<size_t>(header_length), '\0');
    read_exact(in, header.data(), header.size(), path, "the header");

    JsonValue document;
    try
    {
        document = parse_json(header);
    }
    catch (const std::runtime_error& error)
    {
        fail(path, std::string("header is not valid JSON: ") + error.what());
    }
    if (!document.is_object())
    {
        fail(path, "header is not a JSON object");
    }

    const uint64_t data_start = 8 + header_length;
    const uint64_t data_size = file_size - data_start;

    WeightFile weights;
    std::vector<unsigned char> scratch;

    for (const auto& member : document.members())
    {
        const std::string& name = member.first;
        // A reserved key holding a string map, not a tensor. A reader that misses this tries to
        // parse "pt" as a dtype.
        if (name == "__metadata__")
        {
            continue;
        }

        const JsonValue& entry = member.second;
        if (!entry.is_object() || !entry.contains("dtype") || !entry.contains("shape") ||
            !entry.contains("data_offsets"))
        {
            fail(path, "tensor \"" + name + "\": entry is missing dtype, shape or data_offsets");
        }

        DType dtype = DType::F32;
        try
        {
            dtype = core::dtype_from_name(entry["dtype"].as_string());
        }
        catch (const std::invalid_argument& error)
        {
            // Loud, never reinterpreted: reading an F16 tensor as BF16 would produce garbage that
            // looks like weights.
            fail(path, "tensor \"" + name + "\": " + error.what());
        }

        const Shape shape = shape_from(entry["shape"], path, name);

        const JsonValue& offsets = entry["data_offsets"];
        if (!offsets.is_array() || offsets.size() != 2)
        {
            fail(path, "tensor \"" + name + "\": data_offsets is not a pair");
        }
        const int64_t begin = offsets[0].as_int();
        const int64_t end = offsets[1].as_int();
        if (begin < 0 || end < begin)
        {
            fail(path, "tensor \"" + name + "\": data_offsets [" + std::to_string(begin) + ", " +
                           std::to_string(end) + "] is not an ascending range");
        }

        const uint64_t claimed = static_cast<uint64_t>(end - begin);
        const size_t needed = core::byte_length(shape, dtype);
        if (claimed != needed)
        {
            fail(path, "tensor \"" + name + "\": claims " + std::to_string(claimed) +
                           " bytes, shape " + shape.to_string() + " of " +
                           core::dtype_name(dtype) + " needs " + std::to_string(needed));
        }
        if (static_cast<uint64_t>(end) > data_size)
        {
            fail(path, "tensor \"" + name + "\": ends at " + std::to_string(end) + ", past the " +
                           std::to_string(data_size) + "-byte data section");
        }

        // Read per tensor rather than slurping the file: the whole of a 1.2 GB model in memory plus
        // its 2.4 GB widened form would be 3.6 GB at peak, for no reason.
        Tensor tensor{shape};
        if (needed > 0)
        {
            scratch.resize(needed);
            in.seekg(static_cast<std::streamoff>(data_start + static_cast<uint64_t>(begin)));
            read_exact(in, reinterpret_cast<char*>(scratch.data()), needed, path,
                       "tensor \"" + name + "\"");

            float* destination = tensor.data();
            if (dtype == DType::F32)
            {
                for (size_t i = 0; i < shape.size(); ++i)
                {
                    uint32_t bits = 0;
                    for (int b = 0; b < 4; ++b)
                    {
                        bits |= static_cast<uint32_t>(scratch[i * 4 + b]) << (8 * b);
                    }
                    destination[i] = std::bit_cast<float>(bits);
                }
            }
            else
            {
                // bf16 is the top 16 bits of the fp32 value: the widening is a shift, and it lives
                // in core (E1.S3.T10) rather than being written again here.
                std::vector<uint16_t> source(shape.size());
                for (size_t i = 0; i < shape.size(); ++i)
                {
                    source[i] = static_cast<uint16_t>(scratch[i * 2]) |
                                static_cast<uint16_t>(static_cast<uint16_t>(scratch[i * 2 + 1]) << 8);
                }
                core::widen_bf16(source.data(), destination, shape.size());
            }
        }

        weights.add(name, std::move(tensor), needed);
    }

    return weights;
}

} // namespace veda::io
