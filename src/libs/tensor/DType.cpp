#include "DType.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace veda::core
{

size_t dtype_size(DType dtype)
{
    switch (dtype)
    {
        case DType::F32:
            return 4;
        case DType::BF16:
            return 2;
    }
    throw std::invalid_argument("dtype_size: unknown DType value");
}

const char* dtype_name(DType dtype)
{
    switch (dtype)
    {
        case DType::F32:
            return "F32";
        case DType::BF16:
            return "BF16";
    }
    throw std::invalid_argument("dtype_name: unknown DType value");
}

DType dtype_from_name(std::string_view name)
{
    if (name == "F32")
    {
        return DType::F32;
    }
    if (name == "BF16")
    {
        return DType::BF16;
    }
    throw std::invalid_argument(
        "dtype_from_name: unsupported dtype \"" + std::string(name) + "\"; Veda reads F32 and BF16 only");
}

size_t byte_length(const Shape& shape, DType dtype)
{
    const size_t elements = shape.size();
    const size_t element_size = dtype_size(dtype);

    // A corrupt header can claim any shape at all, so the multiplication is checked rather than
    // trusted: an overflow here would silently under-allocate.
    if (elements != 0 && element_size > std::numeric_limits<size_t>::max() / elements)
    {
        throw std::overflow_error(
            "byte_length: shape " + shape.to_string() + " of " + dtype_name(dtype) +
            " overflows size_t");
    }

    return elements * element_size;
}

} // namespace veda::core
