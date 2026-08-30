#ifndef VEDA_TENSOR_FILE_H
#define VEDA_TENSOR_FILE_H

#include "Tensor.h"

#include <string>

// The golden-file format: one fp32 tensor per file, little-endian throughout.
//
//   offset  size        contents
//   0       8 bytes     magic "VEDATNSR"
//   8       4 bytes     uint32   dtype code (0 = F32; nothing else is written)
//   12      4 bytes     uint32   rank
//   16      8*rank      uint64   dimensions, outermost first
//   ...     4*numel     float32  data, contiguous row-major
//
// Deliberately not safetensors: E4 will read that format, and a golden file that shares code with
// the loader under test is a harness that can agree with a bug. This one is written by fifteen
// lines of dependency-free Python (tools/veda_tensor.py, T4).
//
// Development tooling: nothing in the veda binary links against this.
namespace veda::testing
{

// Reads a tensor written in the format above. Throws std::runtime_error naming the path for a
// missing file, a wrong magic, an unknown dtype, or a body whose length disagrees with the header.
core::Tensor read_tensor(const std::string& path);

// Writes any tensor — including a non-contiguous view, which is stored in index order, because a
// dumped tensor is a value and strides are not part of the format.
void write_tensor(const std::string& path, const core::Tensor& tensor);

} // namespace veda::testing

#endif //VEDA_TENSOR_FILE_H
