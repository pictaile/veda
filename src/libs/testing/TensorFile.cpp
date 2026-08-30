#include "TensorFile.h"

#include "Shape.h"
#include "Storage.h"
#include "Walk.h"

#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace veda::testing
{

using core::Shape;
using core::Tensor;

namespace
{

constexpr std::array<char, 8> magic = {'V', 'E', 'D', 'A', 'T', 'N', 'S', 'R'};
constexpr uint32_t dtype_f32 = 0;

// The header is written field by field with fixed-width types and explicit byte order — never by
// memcpy-ing a struct, whose padding is a portability trap that costs an hour to find.
void put_u32(std::ostream& out, uint32_t value)
{
    for (int shift = 0; shift < 32; shift += 8)
    {
        const char byte = static_cast<char>((value >> shift) & 0xFF);
        out.write(&byte, 1);
    }
}

void put_u64(std::ostream& out, uint64_t value)
{
    for (int shift = 0; shift < 64; shift += 8)
    {
        const char byte = static_cast<char>((value >> shift) & 0xFF);
        out.write(&byte, 1);
    }
}

void put_f32(std::ostream& out, float value)
{
    put_u32(out, std::bit_cast<uint32_t>(value));
}

[[noreturn]] void fail(const std::string& path, const std::string& reason)
{
    throw std::runtime_error("tensor file \"" + path + "\": " + reason);
}

uint32_t take_u32(std::istream& in, const std::string& path)
{
    uint32_t value = 0;
    for (int shift = 0; shift < 32; shift += 8)
    {
        char byte = 0;
        if (!in.read(&byte, 1))
        {
            fail(path, "truncated header");
        }
        value |= static_cast<uint32_t>(static_cast<unsigned char>(byte)) << shift;
    }
    return value;
}

uint64_t take_u64(std::istream& in, const std::string& path)
{
    uint64_t value = 0;
    for (int shift = 0; shift < 64; shift += 8)
    {
        char byte = 0;
        if (!in.read(&byte, 1))
        {
            fail(path, "truncated header");
        }
        value |= static_cast<uint64_t>(static_cast<unsigned char>(byte)) << shift;
    }
    return value;
}

} // namespace

void write_tensor(const std::string& path, const Tensor& tensor)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        fail(path, "cannot be opened for writing");
    }

    out.write(magic.data(), magic.size());
    put_u32(out, dtype_f32);
    put_u32(out, static_cast<uint32_t>(tensor.rank()));
    for (const size_t dimension : tensor.shape().dims())
    {
        put_u64(out, static_cast<uint64_t>(dimension));
    }

    // Walked through the strides, so a transposed or sliced view is stored in index order.
    const Shape& shape = tensor.shape();
    std::vector<size_t> index(shape.rank(), 0);
    for (size_t i = 0; i < shape.size(); ++i)
    {
        put_f32(out, ops::detail::read(tensor, index));
        ops::detail::advance(index, shape);
    }

    if (!out)
    {
        fail(path, "write failed");
    }
}

Tensor read_tensor(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        fail(path, "cannot be opened");
    }

    // The magic is checked before anything else: a truncated or unrelated file must fail loudly
    // rather than produce a tensor of noise.
    std::array<char, 8> header{};
    if (!in.read(header.data(), header.size()) || header != magic)
    {
        fail(path, "is not a veda tensor file (bad magic)");
    }

    const uint32_t dtype = take_u32(in, path);
    if (dtype != dtype_f32)
    {
        fail(path, "unsupported dtype code " + std::to_string(dtype) + "; only F32 (0) is read");
    }

    const uint32_t rank = take_u32(in, path);
    std::vector<size_t> dims;
    dims.reserve(rank);
    for (uint32_t k = 0; k < rank; ++k)
    {
        dims.push_back(static_cast<size_t>(take_u64(in, path)));
    }

    const Shape shape(std::move(dims));
    Tensor tensor{shape};

    // The header states the element count, so the body length is known and can be checked against
    // what the file actually holds. A file whose length disagrees with its header is corrupt.
    const std::streampos body_start = in.tellg();
    in.seekg(0, std::ios::end);
    const std::streamoff available = in.tellg() - body_start;
    const std::streamoff needed = static_cast<std::streamoff>(shape.size()) * 4;
    if (available != needed)
    {
        fail(path, "body is " + std::to_string(available) + " bytes, header implies " +
                       std::to_string(needed) + " for shape " + shape.to_string());
    }
    in.seekg(body_start);

    float* destination = tensor.data();
    for (size_t i = 0; i < shape.size(); ++i)
    {
        destination[i] = std::bit_cast<float>(take_u32(in, path));
    }

    return tensor;
}

} // namespace veda::testing
