#ifndef VEDA_DTYPE_H
#define VEDA_DTYPE_H

#include "Shape.h"

#include <cstddef>
#include <string_view>
#include <vector>

namespace veda::core
{

// What the bytes in a weight file mean.
//
// Veda computes in fp32 only (AD1), so this type exists at the I/O boundary and nowhere else: the
// safetensors header names a dtype as a string, and the loader needs its byte size to locate a
// tensor in the file and to refuse formats it cannot handle. No op, no layer and no model code
// ever branches on a DType — that is what keeps the kernel count from multiplying.
//
// The enum stays minimal on purpose. Adding F16 or I8 "for later" would add code paths that are
// never exercised and therefore wrong the first time they are needed.
enum class DType
{
    F32,
    BF16,
};

// Bytes occupied by one element.
size_t dtype_size(DType dtype);

// The name safetensors uses, e.g. "BF16". Also used in error messages.
const char* dtype_name(DType dtype);

// Parses a safetensors dtype string. The only place the name <-> enum mapping exists.
// Throws std::invalid_argument, naming the offending string, for anything unsupported —
// silently reading an F16 tensor as BF16 would produce garbage that looks like weights.
DType dtype_from_name(std::string_view name);

// How many bytes a tensor of this shape and dtype occupies in a file. The loader compares this
// against the extent the header claims; a mismatch means a corrupt header.
// Throws std::overflow_error if a (corrupt) shape makes the product unrepresentable.
size_t byte_length(const Shape& shape, DType dtype);

// What the loader reads raw file bytes into, before widening them into a Tensor (T10).
using ByteBuffer = std::vector<std::byte>;

} // namespace veda::core

#endif //VEDA_DTYPE_H
