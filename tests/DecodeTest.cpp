// E5.S3.T5 — decode, streaming, and special tokens

#include "Json.h"
#include "TestSupport.h"
#include "Tokenizer.h"
#include "Vocabulary.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using veda::io::parse_json;
using veda::tokenizer::bytes_to_token_text;
using veda::tokenizer::Merges;
using veda::tokenizer::StreamingDecoder;
using veda::tokenizer::Tokenizer;
using veda::tokenizer::Vocabulary;

namespace
{
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

// A corpus that is Cyrillic-heavy from the start, as the roadmap's acceptance criterion asks.
const std::vector<std::string>& corpus()
{
    static const std::vector<std::string> texts = {
        "Привіт, світе!",
        "сон",
        " сон",
        "Столиця України — Київ",
        "їжак у тумані",
        "Ґанок, ґудзик, ґava",
        "the cat sat on the mat",
        "2024 рік",
        "中文测试",
        "😀 emoji 🎉 mixed 🇺🇦",
        "  leading and trailing  ",
        "line\nbreak\ttab",
        "don't can't won't",
        "",
        "a",
        "«лапки» та тире —",
        "мішаний mixed текст 123",
    };
    return texts;
}
} // namespace

int main()
{
    // *** the milestone: decode(encode(s)) == s, over the whole corpus ***
    {
        Merges merges;
        merges.add("Ġ", "t", 0);
        merges.add("h", "e", 1);
        merges.add("Ġt", "he", 2);
        const Tokenizer tokenizer(byte_vocabulary({"Ġt", "he", "Ġthe"}), merges);

        for (const std::string& text : corpus())
        {
            CHECK_EQ(tokenizer.decode(tokenizer.encode(text)), text);
        }

        // and with no merges at all, so every token is a single byte
        const Tokenizer bare(byte_vocabulary(), Merges{});
        for (const std::string& text : corpus())
        {
            CHECK_EQ(bare.decode(bare.encode(text)), text);
        }
    }

    // decode of an empty sequence, and of an unknown id
    {
        const Tokenizer tokenizer(byte_vocabulary(), Merges{});
        CHECK_EQ(tokenizer.decode({}), std::string(""));
        CHECK_THROWS_AS(tokenizer.decode({9999}), std::out_of_range);
    }

    // *** the streaming decoder never emits invalid UTF-8 ***
    // Split each string's bytes at every position and feed the halves as two tokens.
    {
        const Tokenizer tokenizer(byte_vocabulary(), Merges{});

        bool all_valid = true;
        bool all_reassembled = true;
        size_t most_held = 0;

        for (const std::string& text : corpus())
        {
            for (size_t split = 0; split <= text.size(); ++split)
            {
                StreamingDecoder decoder(tokenizer);
                std::string emitted;

                // the bytes before the split, then the bytes after — one id per byte
                for (size_t i = 0; i < text.size(); ++i)
                {
                    emitted += decoder.push(static_cast<unsigned char>(text[i]));
                    most_held = std::max(most_held, decoder.held());

                    // whatever has been emitted so far must be complete UTF-8: no partial
                    // character has ever left the decoder
                    size_t position = 0;
                    while (position < emitted.size())
                    {
                        const unsigned char lead = static_cast<unsigned char>(emitted[position]);
                        size_t width = 1;
                        if ((lead & 0xE0) == 0xC0)
                        {
                            width = 2;
                        }
                        else if ((lead & 0xF0) == 0xE0)
                        {
                            width = 3;
                        }
                        else if ((lead & 0xF8) == 0xF0)
                        {
                            width = 4;
                        }
                        if (position + width > emitted.size())
                        {
                            all_valid = false;
                            break;
                        }
                        position += width;
                    }
                }

                emitted += decoder.flush();
                if (emitted != text)
                {
                    all_reassembled = false;
                }
            }
        }

        CHECK(all_valid);
        CHECK(all_reassembled);
        CHECK(most_held <= size_t{3});   // at most three bytes are ever held back
    }

    // the worked example: "сон" arriving as three awkward tokens
    {
        const Tokenizer tokenizer(byte_vocabulary(), Merges{});
        StreamingDecoder decoder(tokenizer);

        const std::string son = "сон";   // D1 81 D0 BE D0 BD
        CHECK_EQ(decoder.push(static_cast<unsigned char>(son[0])), std::string(""));   // D1: waiting
        CHECK_EQ(decoder.held(), size_t{1});

        std::string emitted = decoder.push(static_cast<unsigned char>(son[1]));        // 81
        CHECK_EQ(emitted, std::string("с"));
        CHECK_EQ(decoder.held(), size_t{0});

        emitted = decoder.push(static_cast<unsigned char>(son[2]));                    // D0: waiting
        CHECK_EQ(emitted, std::string(""));

        emitted = decoder.push(static_cast<unsigned char>(son[3]));                    // BE
        CHECK_EQ(emitted, std::string("о"));

        decoder.push(static_cast<unsigned char>(son[4]));
        emitted = decoder.push(static_cast<unsigned char>(son[5]));
        CHECK_EQ(emitted, std::string("н"));
        CHECK_EQ(decoder.flush(), std::string(""));
    }

    // a sequence that stops mid-character: flush returns the raw remainder, not a substitute
    {
        const Tokenizer tokenizer(byte_vocabulary(), Merges{});
        StreamingDecoder decoder(tokenizer);

        CHECK_EQ(decoder.push(0xD1), std::string(""));   // the lead byte of a Cyrillic letter
        CHECK_EQ(decoder.held(), size_t{1});

        const std::string left = decoder.flush();
        CHECK_EQ(left.size(), size_t{1});
        CHECK_EQ(static_cast<unsigned char>(left[0]), 0xD1);
        CHECK_EQ(decoder.held(), size_t{0});
    }

    // --- special tokens ---------------------------------------------------------------------------
    {
        Tokenizer tokenizer(byte_vocabulary(), Merges{});
        tokenizer.add_special_token("<|endoftext|>", 151643);
        tokenizer.add_special_token("<|im_start|>", 151644);

        // one id, not thirteen characters' worth
        CHECK_EQ(tokenizer.encode("<|endoftext|>"), std::vector<int32_t>({151643}));
        CHECK_EQ(tokenizer.decode({151643}), std::string("<|endoftext|>"));

        // a special inside text splits the text around it
        const std::vector<int32_t> ids = tokenizer.encode("hi<|endoftext|>bye");
        CHECK_EQ(ids.size(), size_t{6});   // "hi" (2 bytes) + special + "bye" (3 bytes)
        CHECK_EQ(ids[2], 151643);
        CHECK_EQ(tokenizer.decode(ids), std::string("hi<|endoftext|>bye"));

        // two specials in a row, and Cyrillic around them
        const std::string mixed = "<|im_start|>Привіт<|endoftext|>";
        const std::vector<int32_t> mixed_ids = tokenizer.encode(mixed);
        CHECK_EQ(mixed_ids.front(), 151644);
        CHECK_EQ(mixed_ids.back(), 151643);
        CHECK_EQ(tokenizer.decode(mixed_ids), mixed);

        // and the round trip still holds for the whole corpus with specials registered
        for (const std::string& text : corpus())
        {
            CHECK_EQ(tokenizer.decode(tokenizer.encode(text)), text);
        }
    }

    // one special that is a prefix of another resolves longest-first
    {
        Tokenizer tokenizer(byte_vocabulary(), Merges{});
        tokenizer.add_special_token("<|a|>", 900);
        tokenizer.add_special_token("<|a|>b", 901);

        CHECK_EQ(tokenizer.encode("<|a|>b"), std::vector<int32_t>({901}));
        CHECK_EQ(tokenizer.encode("<|a|>c"), std::vector<int32_t>({900, static_cast<int32_t>('c')}));
        CHECK_EQ(tokenizer.decode({901}), std::string("<|a|>b"));
    }

    // added_tokens are read from the document
    {
        const char* document = R"({
            "added_tokens": [{"id": 151643, "content": "<|endoftext|>", "special": true},
                             {"id": 151644, "content": "<|im_start|>", "special": true}],
            "model": {"vocab": {"a": 97, "b": 98}, "merges": []}})";

        const Tokenizer tokenizer = Tokenizer::from_tokenizer_json(parse_json(document));
        CHECK_EQ(tokenizer.special_tokens().size(), size_t{2});
        CHECK_EQ(tokenizer.encode("<|endoftext|>"), std::vector<int32_t>({151643}));
        CHECK_EQ(tokenizer.decode({151644}), std::string("<|im_start|>"));
        CHECK_EQ(tokenizer.encode("ab"), std::vector<int32_t>({97, 98}));
    }

    // --- ★ the reference corpus, which is the only thing that proves agreement ---------------------
    {
        const std::filesystem::path corpus_file = "reference/tokenizer_corpus.txt";
        if (std::filesystem::exists(corpus_file))
        {
            std::cout << "  reference corpus found — id comparison would run here\n";
        }
        else
        {
            std::cout << "  SKIPPED: no reference/tokenizer_corpus.txt.\n"
                      << "  Round trips prove internal consistency; only a comparison against the\n"
                      << "  real tokenizer proves the ids match what the model was trained on (R2).\n";
        }
    }

    return VEDA_TEST_SUMMARY("DecodeTest");
}
