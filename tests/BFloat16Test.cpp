// E1.S3.T10 — bf16 -> fp32 widening

#include "BFloat16.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using veda::core::bf16_to_f32;
using veda::core::Shape;
using veda::core::Tensor;
using veda::core::widen_bf16;

int main()
{
    // the table from the task — compared with ==, never a tolerance: the widening is exact and a
    // tolerance would hide exactly the bug this function can have
    CHECK_EQ(bf16_to_f32(0x0000), 0.0f);
    CHECK_EQ(bf16_to_f32(0x3F80), 1.0f);
    CHECK_EQ(bf16_to_f32(0xBF80), -1.0f);
    CHECK_EQ(bf16_to_f32(0x4000), 2.0f);
    CHECK_EQ(bf16_to_f32(0x4049), 3.140625f);   // pi, as much of it as bf16 holds
    CHECK_EQ(bf16_to_f32(0x3F00), 0.5f);
    CHECK_EQ(bf16_to_f32(0xC000), -2.0f);

    // it is a compile-time function: these are checked by the compiler, not at run time
    static_assert(bf16_to_f32(0x3F80) == 1.0f);
    static_assert(bf16_to_f32(0xBF80) == -1.0f);
    static_assert(bf16_to_f32(0x0000) == 0.0f);

    // the trap this function exists to avoid: converting the integer instead of reinterpreting it
    CHECK(bf16_to_f32(0x3F80) != 1065353216.0f);
    CHECK_EQ(static_cast<float>(static_cast<uint32_t>(0x3F80) << 16), 1065353216.0f);

    // special values need no special-case code
    {
        CHECK(std::isinf(bf16_to_f32(0x7F80)));
        CHECK(bf16_to_f32(0x7F80) > 0.0f);
        CHECK(std::isinf(bf16_to_f32(0xFF80)));
        CHECK(bf16_to_f32(0xFF80) < 0.0f);
        CHECK(std::isnan(bf16_to_f32(0x7FC0)));
        CHECK(std::isnan(bf16_to_f32(0xFFC0)));

        // negative zero stays negative
        const float negative_zero = bf16_to_f32(0x8000);
        CHECK_EQ(negative_zero, 0.0f);              // -0.0 == 0.0 by IEEE comparison
        CHECK(std::signbit(negative_zero));         // but the sign bit survived
        CHECK(!std::signbit(bf16_to_f32(0x0000)));
        CHECK_EQ(std::bit_cast<uint32_t>(negative_zero), uint32_t{0x80000000});
    }

    // exhaustive: every one of the 65536 bf16 patterns widens to exactly its bits shifted left 16
    {
        size_t exact = 0;
        for (uint32_t b = 0; b <= 0xFFFF; ++b)
        {
            const uint16_t bits = static_cast<uint16_t>(b);
            if (std::bit_cast<uint32_t>(bf16_to_f32(bits)) == (static_cast<uint32_t>(bits) << 16))
            {
                ++exact;
            }
        }
        CHECK_EQ(exact, size_t{65536});
    }

    // the whole-array path
    {
        const uint16_t src[4] = {0x0000, 0x3F80, 0xBF80, 0x4000};
        float dst[4] = {9.0f, 9.0f, 9.0f, 9.0f};

        widen_bf16(src, dst, 4);
        CHECK_EQ(dst[0], 0.0f);
        CHECK_EQ(dst[1], 1.0f);
        CHECK_EQ(dst[2], -1.0f);
        CHECK_EQ(dst[3], 2.0f);
    }

    // n == 0 writes nothing
    {
        const uint16_t src[1] = {0x3F80};
        float dst[1] = {42.0f};
        widen_bf16(src, dst, 0);
        CHECK_EQ(dst[0], 42.0f);
        widen_bf16(nullptr, nullptr, 0);   // legal: nothing to do
    }

    // the loop covers the whole range it is given and nothing beyond it
    {
        std::vector<uint16_t> src(1000);
        for (size_t i = 0; i < src.size(); ++i)
        {
            src[i] = static_cast<uint16_t>(0x3F80 + i);   // 1.0 upwards
        }
        std::vector<float> dst(src.size() + 1, -1.0f);

        widen_bf16(src.data(), dst.data(), src.size());
        CHECK_EQ(dst.front(), 1.0f);
        CHECK_EQ(dst[999], bf16_to_f32(static_cast<uint16_t>(0x3F80 + 999)));
        CHECK_EQ(dst.back(), -1.0f);   // the guard element past the end is untouched
    }

    // widening straight into a tensor's storage, which is how the loader will use it (E4).
    // The destination is allocated once, by the tensor; this function allocates nothing.
    {
        const Shape shape({2, 3});
        Tensor weights{shape};
        const std::vector<uint16_t> raw = {0x3F80, 0xC000, 0x0000, 0x4049, 0xBF80, 0x4000};

        widen_bf16(raw.data(), weights.data(), shape.size());

        CHECK_EQ(weights.at({0, 0}), 1.0f);
        CHECK_EQ(weights.at({0, 1}), -2.0f);
        CHECK_EQ(weights.at({0, 2}), 0.0f);
        CHECK_EQ(weights.at({1, 0}), 3.140625f);
        CHECK_EQ(weights.at({1, 1}), -1.0f);
        CHECK_EQ(weights.at({1, 2}), 2.0f);

        // and the views built on top of it read the same numbers
        CHECK_EQ(weights.transpose(0, 1).at({0, 1}), 3.140625f);
        CHECK_EQ(weights.reshape(Shape({6})).at({3}), 3.140625f);
        CHECK_EQ(weights.slice(0, 1, 1).at({0, 0}), 3.140625f);
    }

    return VEDA_TEST_SUMMARY("BFloat16Test");
}
