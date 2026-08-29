#ifndef VEDA_BFLOAT16_H
#define VEDA_BFLOAT16_H

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace veda::core
{

// Widens one bf16 value to fp32.
//
// A bf16 value is bit-identical to the top 16 bits of the equivalent fp32 value (same sign, same
// 8-bit exponent, a truncated mantissa), so widening is a left shift of 16 with zeros arriving in
// the low half. It is exact for every input, and zero, negative zero, infinities and NaN all fall
// out of the shift with no special handling — an `if` here would mean the mental model is off.
//
// The bits must be *reinterpreted*, not converted: `float f = bits;` would compute the value
// 1065353216.0f where 1.0f is meant. std::bit_cast is the C++20 tool for that — constexpr, no
// aliasing hazard, and it compiles to nothing. A union or a reinterpret_cast through a pointer
// would violate strict aliasing even where they happen to work.
//
// Byte order is settled by the load that produced the uint16_t, not here; every platform Veda
// targets is little-endian throughout, so no swap is needed.
constexpr float bf16_to_f32(uint16_t bits) noexcept
{
    return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}

// Widens n values from src into the caller-allocated dst. Allocates nothing: every weight in the
// model passes through this loop exactly once at load, and it is memory-bandwidth-bound.
inline void widen_bf16(const uint16_t* src, float* dst, size_t n) noexcept
{
    assert((n == 0 || (src != nullptr && dst != nullptr)) && "widen_bf16: null buffer");

    for (size_t i = 0; i < n; ++i)
    {
        dst[i] = bf16_to_f32(src[i]);
    }
}

} // namespace veda::core

#endif //VEDA_BFLOAT16_H
