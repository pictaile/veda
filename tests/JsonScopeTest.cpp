// E4.S1.T1 — Scoping the JSON reader (a Learn task; the parser arrives in T2)
//
// The two things worth settling before writing a parser: what a \uXXXX escape actually means, and
// how a codepoint becomes UTF-8. Both are where the bugs live, and both are checkable on paper.

#include "TestSupport.h"

#include <cstdint>
#include <string>
#include <vector>

namespace
{
// The four ranges of UTF-8. Getting a boundary wrong produces text that passes ASCII tests and
// breaks on the first Cyrillic string — exactly the case E5 cares about most.
std::string to_utf8(uint32_t codepoint)
{
    std::string out;
    if (codepoint < 0x80)
    {
        out += static_cast<char>(codepoint);
    }
    else if (codepoint < 0x800)
    {
        out += static_cast<char>(0xC0 | (codepoint >> 6));
        out += static_cast<char>(0x80 | (codepoint & 0x3F));
    }
    else if (codepoint < 0x10000)
    {
        out += static_cast<char>(0xE0 | (codepoint >> 12));
        out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (codepoint & 0x3F));
    }
    else
    {
        out += static_cast<char>(0xF0 | (codepoint >> 18));
        out += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (codepoint & 0x3F));
    }
    return out;
}

// A \uXXXX escape is a UTF-16 code unit. Anything outside the basic plane is written as two of
// them, and they only mean something together.
uint32_t combine_surrogates(uint32_t high, uint32_t low)
{
    return 0x10000 + (high - 0xD800) * 0x400 + (low - 0xDC00);
}

bool is_high_surrogate(uint32_t unit) { return unit >= 0xD800 && unit <= 0xDBFF; }
bool is_low_surrogate(uint32_t unit) { return unit >= 0xDC00 && unit <= 0xDFFF; }
} // namespace

int main()
{
    // one byte per character below 0x80 — the ASCII range, where nothing interesting happens
    CHECK_EQ(to_utf8(0x0041), std::string("A"));
    CHECK_EQ(to_utf8(0x0041).size(), size_t{1});

    // Cyrillic lives in the two-byte range, which is why byte-level tokenization treats it
    // completely differently from Latin (E5, risk R2)
    CHECK_EQ(to_utf8(0x0416), std::string("Ж"));
    CHECK_EQ(to_utf8(0x0416).size(), size_t{2});
    CHECK_EQ(to_utf8(0x0441), std::string("с"));

    // CJK is three bytes
    CHECK_EQ(to_utf8(0x4E2D), std::string("中"));
    CHECK_EQ(to_utf8(0x4E2D).size(), size_t{3});

    // and the boundaries themselves, where an off-by-one would hide
    CHECK_EQ(to_utf8(0x007F).size(), size_t{1});
    CHECK_EQ(to_utf8(0x0080).size(), size_t{2});
    CHECK_EQ(to_utf8(0x07FF).size(), size_t{2});
    CHECK_EQ(to_utf8(0x0800).size(), size_t{3});
    CHECK_EQ(to_utf8(0xFFFF).size(), size_t{3});
    CHECK_EQ(to_utf8(0x10000).size(), size_t{4});

    // a surrogate pair: "😀" is one emoji, not two characters
    {
        const uint32_t high = 0xD83D;
        const uint32_t low = 0xDE00;
        CHECK(is_high_surrogate(high));
        CHECK(is_low_surrogate(low));
        CHECK(!is_high_surrogate(low));

        const uint32_t combined = combine_surrogates(high, low);
        CHECK_EQ(combined, uint32_t{0x1F600});
        CHECK_EQ(to_utf8(combined), std::string("😀"));
        CHECK_EQ(to_utf8(combined).size(), size_t{4});

        // decoding the halves independently produces two codepoints that are not valid on their
        // own — the failure mode this whole check exists to prevent
        CHECK(to_utf8(high) + to_utf8(low) != to_utf8(combined));
        CHECK_EQ((to_utf8(high) + to_utf8(low)).size(), size_t{6});
    }

    // the surrogate range is reserved: no character lives there, so a lone half is always an error
    {
        for (const uint32_t unit : {0xD800u, 0xDBFFu, 0xDC00u, 0xDFFFu})
        {
            CHECK(is_high_surrogate(unit) || is_low_surrogate(unit));
        }
        CHECK(!is_high_surrogate(0xD7FF) && !is_low_surrogate(0xD7FF));
        CHECK(!is_high_surrogate(0xE000) && !is_low_surrogate(0xE000));
    }

    // why a naive parser is quadratic: copying the remaining input per token costs O(n) each time
    {
        const size_t document = 7'000'000;      // tokenizer.json, roughly
        const size_t tokens = 300'000;

        const size_t linear = document;                 // one pass, an index into one buffer
        const size_t naive = tokens * (document / 2);   // an average-length substring per token
        CHECK(naive / linear > 10'000);                 // four orders of magnitude, on the real file
    }

    return VEDA_TEST_SUMMARY("JsonScopeTest");
}
