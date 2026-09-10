// E5.S2.T3 (merges and pre-tokenization) and T4 (encode)

#include "Json.h"
#include "TestSupport.h"
#include "Tokenizer.h"
#include "Vocabulary.h"

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using veda::io::parse_json;
using veda::tokenizer::bytes_to_token_text;
using veda::tokenizer::Merges;
using veda::tokenizer::pre_tokenize;
using veda::tokenizer::Tokenizer;
using veda::tokenizer::Vocabulary;

namespace
{
// A vocabulary with every byte plus whatever extra tokens a test needs.
Vocabulary byte_vocabulary(const std::vector<std::string>& extra = {})
{
    Vocabulary vocabulary;
    for (int b = 0; b < 256; ++b)
    {
        vocabulary.add(bytes_to_token_text(std::string(1, static_cast<char>(b))), b);
    }
    int32_t next = 256;
    for (const std::string& token : extra)
    {
        vocabulary.add(token, next++);
    }
    return vocabulary;
}

std::string joined(const std::vector<std::string>& pieces)
{
    return std::accumulate(pieces.begin(), pieces.end(), std::string{});
}
} // namespace

int main()
{
    // --- pre-tokenization ------------------------------------------------------------------------
    {
        // the leading space joins the following word
        const auto cat = pre_tokenize("the cat");
        CHECK_EQ(cat.size(), size_t{2});
        CHECK_EQ(cat[0], std::string("the"));
        CHECK_EQ(cat[1], std::string(" cat"));

        // digits, one at a time: a number never becomes a single token
        const auto year = pre_tokenize("2024 рік");
        CHECK_EQ(year.size(), size_t{5});
        CHECK_EQ(year[0], std::string("2"));
        CHECK_EQ(year[3], std::string("4"));
        CHECK_EQ(year[4], std::string(" рік"));

        // punctuation is its own run
        const auto greeting = pre_tokenize("Привіт!");
        CHECK_EQ(greeting.size(), size_t{2});
        CHECK_EQ(greeting[0], std::string("Привіт"));
        CHECK_EQ(greeting[1], std::string("!"));

        // newline runs stay together
        const auto lines = pre_tokenize("a\n\nb");
        CHECK_EQ(lines.size(), size_t{3});
        CHECK_EQ(lines[1], std::string("\n\n"));

        // *** the pieces concatenate back to the input, exactly — no trimming, no substitution ***
        for (const char* text : {"the cat", "2024 рік", "Привіт, світе!", "a\n\nb", "  spaced  out ",
                                 "don't", "сон", " сон", "😀 emoji", "", "\t\t", "ROMEO: Hi!"})
        {
            CHECK_EQ(joined(pre_tokenize(text)), std::string(text));
        }
    }

    // --- the hand-traced merge from T3 -------------------------------------------------------------
    // merges:  "Ġ t" -> 0,  "h e" -> 1,  "Ġt he" -> 2
    {
        Merges merges;
        merges.add("Ġ", "t", 0);
        merges.add("h", "e", 1);
        merges.add("Ġt", "he", 2);

        CHECK_EQ(*merges.rank("Ġ", "t"), size_t{0});
        CHECK_EQ(*merges.rank("Ġt", "he"), size_t{2});
        CHECK(!merges.rank("t", "h").has_value());

        const Tokenizer tokenizer(byte_vocabulary({"Ġt", "he", "Ġthe"}), merges);
        const std::vector<int32_t> ids = tokenizer.encode(" the");

        // Ġ t h e -> Ġt h e -> Ġt he -> Ġthe
        CHECK_EQ(ids.size(), size_t{1});
        CHECK_EQ(ids[0], 258);   // 256 = "Ġt", 257 = "he", 258 = "Ġthe"
        CHECK_EQ(tokenizer.vocabulary().text_of(ids[0]), std::string("Ġthe"));
        CHECK_EQ(tokenizer.vocabulary().bytes_of(ids[0]), std::string(" the"));
    }

    // *** BPE is not longest-match: the constructed case where they disagree ***
    // vocabulary has "ab", "cd" and "abc"; merges are (a,b)=0, (c,d)=1, (ab,c)=2
    {
        Merges merges;
        merges.add("a", "b", 0);
        merges.add("c", "d", 1);
        merges.add("ab", "c", 2);

        const Tokenizer tokenizer(byte_vocabulary({"ab", "cd", "abc"}), merges);
        const std::vector<int32_t> ids = tokenizer.encode("abcd");

        // BPE: lowest rank first -> ab c d -> ab cd -> stop
        CHECK_EQ(ids.size(), size_t{2});
        CHECK_EQ(tokenizer.vocabulary().text_of(ids[0]), std::string("ab"));
        CHECK_EQ(tokenizer.vocabulary().text_of(ids[1]), std::string("cd"));

        // longest-match would have taken "abc" first and produced [abc, d] — different ids
        CHECK(tokenizer.vocabulary().text_of(ids[0]) != std::string("abc"));
        CHECK_EQ(*tokenizer.vocabulary().id_of("abc"), 258);   // it exists, and BPE did not use it
    }

    // merging takes every occurrence of the winning pair before re-scanning
    {
        Merges merges;
        merges.add("a", "b", 0);
        const Tokenizer tokenizer(byte_vocabulary({"ab"}), merges);

        const std::vector<int32_t> ids = tokenizer.encode("abab");
        CHECK_EQ(ids.size(), size_t{2});
        CHECK_EQ(tokenizer.vocabulary().text_of(ids[0]), std::string("ab"));
        CHECK_EQ(tokenizer.vocabulary().text_of(ids[1]), std::string("ab"));
    }

    // *** "сон" and " сон" produce different ids — the epic's headline property ***
    {
        Merges merges;
        const Tokenizer tokenizer(byte_vocabulary(), merges);

        const std::vector<int32_t> bare = tokenizer.encode("сон");
        const std::vector<int32_t> spaced = tokenizer.encode(" сон");

        CHECK_EQ(bare.size(), size_t{6});      // six bytes, no merges available
        CHECK_EQ(spaced.size(), size_t{7});    // and the space is a seventh
        CHECK(bare != spaced);
        CHECK_EQ(spaced[0], 0x20);             // the space, as its own byte token

        // the Cyrillic penalty, in a number: the same meaning costs more ids
        CHECK_EQ(tokenizer.encode("cat").size(), size_t{3});
        CHECK_EQ(tokenizer.encode("кіт").size(), size_t{6});
    }

    // a string with no applicable merge falls back to bytes, one id each
    {
        Merges merges;
        merges.add("x", "y", 0);
        const Tokenizer tokenizer(byte_vocabulary({"xy"}), merges);

        const std::vector<int32_t> ids = tokenizer.encode("qwe");
        CHECK_EQ(ids.size(), size_t{3});
        CHECK_EQ(ids[0], static_cast<int32_t>('q'));
        CHECK_EQ(ids[2], static_cast<int32_t>('e'));

        // and one emoji is four byte ids
        CHECK_EQ(tokenizer.encode("😀").size(), size_t{4});
        CHECK_EQ(tokenizer.encode("").size(), size_t{0});
    }

    // merges load from both spellings the format uses
    {
        const char* as_strings = R"({"model": {"vocab": {"a": 0}, "merges": ["Ġ t", "h e"]}})";
        const char* as_arrays = R"({"model": {"vocab": {"a": 0}, "merges": [["Ġ", "t"], ["h", "e"]]}})";

        const Merges from_strings = Merges::from_tokenizer_json(parse_json(as_strings));
        const Merges from_arrays = Merges::from_tokenizer_json(parse_json(as_arrays));

        CHECK_EQ(from_strings.size(), size_t{2});
        CHECK_EQ(from_arrays.size(), size_t{2});
        CHECK_EQ(*from_strings.rank("Ġ", "t"), size_t{0});
        CHECK_EQ(*from_arrays.rank("Ġ", "t"), size_t{0});
        CHECK_EQ(*from_arrays.rank("h", "e"), size_t{1});

        CHECK_THROWS_AS(Merges::from_tokenizer_json(parse_json(R"({"model": {}})")),
                        std::runtime_error);
        CHECK_THROWS_AS(
            Merges::from_tokenizer_json(parse_json(R"({"model": {"merges": ["noseparator"]}})")),
            std::runtime_error);
    }

    // a whole tokenizer from one document
    {
        const char* document = R"({"model": {"type": "BPE",
            "vocab": {"Ġ": 220, "t": 116, "h": 104, "e": 101, "Ġt": 256, "he": 257, "Ġthe": 258},
            "merges": ["Ġ t", "h e", "Ġt he"]}})";

        const Tokenizer tokenizer = Tokenizer::from_tokenizer_json(parse_json(document));
        CHECK_EQ(tokenizer.merges().size(), size_t{3});
        CHECK_EQ(tokenizer.encode(" the"), std::vector<int32_t>({258}));
        CHECK_EQ(tokenizer.encode("the"), std::vector<int32_t>({116, 257}));   // "t" + "he"
    }

    // a symbol the vocabulary does not know means a broken tokenizer.json
    {
        Vocabulary partial;
        partial.add(bytes_to_token_text("a"), 0);
        const Tokenizer tokenizer(partial, Merges{});

        CHECK_EQ(tokenizer.encode("a"), std::vector<int32_t>({0}));
        CHECK_THROWS_AS(tokenizer.encode("b"), std::runtime_error);
        try
        {
            (void)tokenizer.encode("b");
        }
        catch (const std::runtime_error& error)
        {
            CHECK(std::string(error.what()).find("not in the vocabulary") != std::string::npos);
        }
    }

    // pieces and ids for a realistic mixed string
    {
        Merges merges;
        const Tokenizer tokenizer(byte_vocabulary(), merges);

        const std::string text = "Привіт, світе! 2024";
        const auto pieces = pre_tokenize(text);
        CHECK_EQ(joined(pieces), text);
        CHECK_EQ(pieces[0], std::string("Привіт"));
        CHECK_EQ(pieces[1], std::string(","));
        CHECK_EQ(pieces[2], std::string(" світе"));
        CHECK_EQ(pieces[3], std::string("!"));

        CHECK_EQ(tokenizer.encode(text).size(), text.size());   // no merges: one id per byte
    }

    return VEDA_TEST_SUMMARY("EncodeTest");
}
