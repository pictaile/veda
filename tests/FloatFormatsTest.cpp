// E1.S3.T8 — fp32 / fp16 / bf16 (a Learn task; the widening itself is implemented in T10)
//
// Confirms the bit patterns the widening relies on: bf16 is literally the top 16 bits of the
// corresponding fp32 value, so widening is a shift and nothing else.

#include "TestSupport.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace
{
uint32_t bits_of(float value)
{
    return std::bit_cast<uint32_t>(value);
}

float float_from(uint32_t bits)
{
    return std::bit_cast<float>(bits);
}

// fp32:  1 sign | 8 exponent | 23 mantissa
uint32_t sign_of(float value) { return bits_of(value) >> 31; }
uint32_t exponent_of(float value) { return (bits_of(value) >> 23) & 0xFF; }
uint32_t mantissa_of(float value) { return bits_of(value) & 0x7FFFFF; }

// What saving an fp32 value as bf16 does: keep the top half, drop the rest.
uint16_t truncate_to_bf16(float value)
{
    return static_cast<uint16_t>(bits_of(value) >> 16);
}

// What T10 will implement: shift left by 16, zeros in the low half.
float widen_bf16(uint16_t bits)
{
    return float_from(static_cast<uint32_t>(bits) << 16);
}
} // namespace

int main()
{
    // fp32 layout: 32 bits, of which 1 sign, 8 exponent, 23 mantissa
    static_assert(sizeof(float) == 4);
    static_assert(std::numeric_limits<float>::is_iec559);

    // 1.0f = 0 01111111 00000000000000000000000 = 0x3F800000
    CHECK_EQ(bits_of(1.0f), uint32_t{0x3F800000});
    CHECK_EQ(sign_of(1.0f), uint32_t{0});
    CHECK_EQ(exponent_of(1.0f), uint32_t{127});   // 0b01111111, the bias itself -> 2^0
    CHECK_EQ(mantissa_of(1.0f), uint32_t{0});

    // -1.0f differs in exactly one bit: the sign
    CHECK_EQ(bits_of(-1.0f), uint32_t{0xBF800000});
    CHECK_EQ(bits_of(-1.0f) ^ bits_of(1.0f), uint32_t{0x80000000});
    CHECK_EQ(sign_of(-1.0f), uint32_t{1});
    CHECK_EQ(exponent_of(-1.0f), exponent_of(1.0f));

    // 2.0 is the same mantissa with the exponent one higher
    CHECK_EQ(exponent_of(2.0f), uint32_t{128});
    CHECK_EQ(mantissa_of(2.0f), uint32_t{0});
    CHECK_EQ(exponent_of(0.5f), uint32_t{126});

    // bf16 is the top 16 bits of fp32 — same sign, same 8-bit exponent, 7 mantissa bits left
    CHECK_EQ(truncate_to_bf16(1.0f), uint16_t{0x3F80});
    CHECK_EQ(truncate_to_bf16(-1.0f), uint16_t{0xBF80});

    // widening is a shift, and it is exact
    CHECK_EQ(widen_bf16(0x3F80), 1.0f);
    CHECK_EQ(widen_bf16(0xBF80), -1.0f);
    CHECK_EQ(bits_of(widen_bf16(0x3F80)), uint32_t{0x3F800000});

    // every value that survives a round trip through bf16 widens back exactly
    for (const float value : {0.0f, 1.0f, -1.0f, 2.0f, 0.5f, -0.25f, 256.0f, 1.5f, -3.0f})
    {
        CHECK_EQ(widen_bf16(truncate_to_bf16(value)), value);
    }

    // pi loses precision going INTO bf16 — that happened when the weights were saved, not here
    {
        const float pi = 3.14159265f;
        CHECK_EQ(bits_of(pi), uint32_t{0x40490FDB});
        CHECK_EQ(truncate_to_bf16(pi), uint16_t{0x4049});
        CHECK_EQ(bits_of(widen_bf16(0x4049)), uint32_t{0x40490000});
        CHECK_EQ(widen_bf16(0x4049), 3.140625f);
        CHECK(std::fabs(widen_bf16(0x4049) - pi) < 0.001f);
    }

    // special values fall out of the shift with no special handling — a sign the approach is right
    {
        CHECK_EQ(widen_bf16(0x0000), 0.0f);
        CHECK_EQ(bits_of(widen_bf16(0x8000)), uint32_t{0x80000000});   // negative zero
        CHECK(std::isnan(widen_bf16(0x7FC0)));
        CHECK(std::isinf(widen_bf16(0x7F80)));                          // +inf
        CHECK(std::isinf(widen_bf16(0xFF80)) && widen_bf16(0xFF80) < 0.0f);
    }

    // the range argument: bf16 keeps fp32's 8-bit exponent, so it reaches the same magnitudes.
    // fp16's 5-bit exponent tops out around 65504 — the reason it overflows where bf16 does not.
    {
        CHECK_EQ(widen_bf16(0x7F00), 1.7014118e38f);       // far past fp16's maximum
        CHECK(widen_bf16(0x7F00) > 65504.0f);
        CHECK(widen_bf16(0x0080) < 1.2e-38f);              // and far below its minimum normal
        CHECK(widen_bf16(0x0080) > 0.0f);
    }

    // what a wrong shift would do: << 8 instead of << 16 produces a finite, plausible-looking
    // number in a completely different magnitude — no crash, no error, just wrong output
    {
        const float wrong = float_from(static_cast<uint32_t>(uint16_t{0x3F80}) << 8);
        CHECK(std::isfinite(wrong));
        CHECK(wrong != 1.0f);
        CHECK(wrong < 1e-30f);   // 1.0 would arrive as a denormal-ish crumb
    }

    return VEDA_TEST_SUMMARY("FloatFormatsTest");
}
