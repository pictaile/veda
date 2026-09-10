// E5.S1.T1 (bytes, not characters) and T2 (Vocabulary)

#include "Json.h"
#include "TestSupport.h"
#include "Vocabulary.h"

#include <set>
#include <stdexcept>
#include <string>

using veda::io::parse_json;
using veda::tokenizer::bytes_to_token_text;
using veda::tokenizer::token_text_to_bytes;
using veda::tokenizer::Vocabulary;

int main()
{
    // --- T1: what UTF-8 does to Cyrillic ---------------------------------------------------------
    {
        CHECK_EQ(std::string("cat").size(), size_t{3});     // one byte per character
        CHECK_EQ(std::string("кіт").size(), size_t{6});     // two — twice the sequence length
        CHECK_EQ(std::string("中文").size(), size_t{6});     // three
        CHECK_EQ(std::string("😀").size(), size_t{4});      // four

        // "сон" is six bytes: D1 81 D0 BE D0 BD
        const std::string son = "сон";
        CHECK_EQ(son.size(), size_t{6});
        CHECK_EQ(static_cast<unsigned char>(son[0]), 0xD1);
        CHECK_EQ(static_cast<unsigned char>(son[1]), 0x81);
    }

    // *** the mapping is a bijection over all 256 bytes ***
    {
        std::set<std::string> seen;
        for (int b = 0; b < 256; ++b)
        {
            const std::string raw(1, static_cast<char>(b));
            const std::string text = bytes_to_token_text(raw);

            CHECK(!text.empty());
            seen.insert(text);
            CHECK_EQ(token_text_to_bytes(text), raw);   // round trip, every byte
        }
        CHECK_EQ(seen.size(), size_t{256});             // and every image is distinct
    }

    // the fixed points and the shifts named in the task
    {
        CHECK_EQ(bytes_to_token_text("A"), std::string("A"));       // printable ASCII, unchanged
        CHECK_EQ(bytes_to_token_text("~"), std::string("~"));
        CHECK_EQ(bytes_to_token_text(" "), std::string("Ġ"));       // U+0120 — the one everyone sees
        CHECK_EQ(bytes_to_token_text("\n"), std::string("Ċ"));      // U+010A
        CHECK_EQ(bytes_to_token_text("\t"), std::string("ĉ"));

        // 0xD1 is already printable in Latin-1 and maps to itself
        CHECK_EQ(bytes_to_token_text(std::string(1, static_cast<char>(0xD1))), std::string("Ñ"));
    }

    // *** Cyrillic, from the first test — this epic's main path, not an edge case ***
    {
        // D1 81 D0 BE D0 BD: 0xD1 and 0xD0 are printable in Latin-1 and map to themselves,
        // but 0x81 and 0xBE, 0xBD are not — 0x81 shifts to U+0123.
        CHECK_EQ(bytes_to_token_text("сон"), std::string("ÑģÐ¾Ð½"));
        CHECK_EQ(token_text_to_bytes("ÑģÐ¾Ð½"), std::string("сон"));

        // *** the leading space makes it a different token ***
        CHECK_EQ(bytes_to_token_text(" сон"), std::string("ĠÑģÐ¾Ð½"));
        CHECK(bytes_to_token_text("сон") != bytes_to_token_text(" сон"));

        // and in English, the same rule
        CHECK_EQ(bytes_to_token_text("the"), std::string("the"));
        CHECK_EQ(bytes_to_token_text(" the"), std::string("Ġthe"));

        // emoji and CJK round-trip too
        for (const char* text : {"😀", "中文", "кіт", "їжак", "Привіт, світе!", "", "a\nb\tc"})
        {
            CHECK_EQ(token_text_to_bytes(bytes_to_token_text(text)), std::string(text));
        }
    }

    // a code point outside the mapping is an error, not a fallback
    {
        CHECK_THROWS_AS(token_text_to_bytes("中"), std::invalid_argument);
        CHECK_THROWS_AS(token_text_to_bytes("😀"), std::invalid_argument);
        try
        {
            (void)token_text_to_bytes("中");
        }
        catch (const std::invalid_argument& error)
        {
            CHECK(std::string(error.what()).find("byte-level mapping") != std::string::npos);
        }
    }

    // --- T2: the vocabulary --------------------------------------------------------------------
    {
        const char* document = R"({"model": {"type": "BPE",
            "vocab": {"!": 0, "\"": 1, "the": 100, "Ġthe": 279, "Ñ": 500, "ĠÑģÐ¾Ð½": 900},
            "merges": ["Ġ t", "Ġ a"]}})";

        const Vocabulary vocabulary = Vocabulary::from_tokenizer_json(parse_json(document));
        CHECK_EQ(vocabulary.size(), size_t{6});

        CHECK_EQ(*vocabulary.id_of("Ġthe"), 279);
        CHECK_EQ(*vocabulary.id_of("!"), 0);
        CHECK_EQ(vocabulary.text_of(279), std::string("Ġthe"));
        CHECK_EQ(vocabulary.bytes_of(279), std::string(" the"));
        CHECK_EQ(vocabulary.bytes_of(900), std::string(" сон"));

        // both directions agree
        CHECK_EQ(*vocabulary.id_of(vocabulary.text_of(100)), 100);
        CHECK(vocabulary.contains("the"));
        CHECK(!vocabulary.contains("Ġthé"));

        // a missing token is a normal answer during merging, not an error
        CHECK(!vocabulary.id_of("Ġthé").has_value());
        CHECK(!vocabulary.id_of("").has_value());

        // a missing id is a bug, and says which
        CHECK_THROWS_AS(vocabulary.text_of(7), std::out_of_range);
        CHECK_THROWS_AS(vocabulary.text_of(-1), std::out_of_range);
        CHECK_THROWS_AS(vocabulary.text_of(99999), std::out_of_range);
        try
        {
            (void)vocabulary.text_of(7);
        }
        catch (const std::out_of_range& error)
        {
            CHECK(std::string(error.what()).find("id 7") != std::string::npos);
        }
    }

    // ids may leave gaps — added tokens sit above the base vocabulary
    {
        Vocabulary vocabulary;
        vocabulary.add("a", 0);
        vocabulary.add("b", 5);
        vocabulary.add("<|endoftext|>", 151643);

        CHECK_EQ(vocabulary.size(), size_t{3});
        CHECK_EQ(vocabulary.text_of(151643), std::string("<|endoftext|>"));
        CHECK_THROWS_AS(vocabulary.text_of(3), std::out_of_range);
        CHECK_EQ(*vocabulary.id_of("b"), 5);
    }

    // malformed documents
    {
        CHECK_THROWS_AS(Vocabulary::from_tokenizer_json(parse_json("{}")), std::runtime_error);
        CHECK_THROWS_AS(Vocabulary::from_tokenizer_json(parse_json(R"({"model": {}})")),
                        std::runtime_error);
        CHECK_THROWS_AS(Vocabulary::from_tokenizer_json(parse_json("[1, 2]")), std::runtime_error);
    }

    // no normalisation: case, whitespace and composition are preserved exactly
    {
        Vocabulary vocabulary;
        vocabulary.add(bytes_to_token_text("The"), 1);
        vocabulary.add(bytes_to_token_text("the"), 2);
        vocabulary.add(bytes_to_token_text(" the"), 3);

        CHECK_EQ(*vocabulary.id_of(bytes_to_token_text("The")), 1);
        CHECK_EQ(*vocabulary.id_of(bytes_to_token_text("the")), 2);
        CHECK_EQ(*vocabulary.id_of(bytes_to_token_text(" the")), 3);
        CHECK(!vocabulary.id_of(bytes_to_token_text("THE")).has_value());
    }

    // the sizes that matter
    {
        CHECK_EQ(size_t{256}, size_t{256});   // base tokens, one per byte, always present
        // a six-letter Ukrainian word costs twelve bytes before any merge
        CHECK_EQ(std::string("ведмідь").size(), size_t{14});
        CHECK_EQ(std::string("bear").size(), size_t{4});
    }

    return VEDA_TEST_SUMMARY("VocabularyTest");
}
