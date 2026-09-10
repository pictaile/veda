#ifndef VEDA_VOCABULARY_H
#define VEDA_VOCABULARY_H

#include "Json.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// The tokenizer works on bytes, not characters.
//
// A vocabulary of Unicode characters would need ~150,000 entries before a single merge and would
// still fail on anything unseen. A vocabulary of bytes needs exactly 256 and can never fail: every
// string, in every language, is a sequence of bytes. An unknown word decomposes into bytes rather
// than becoming <unk>.
//
// What that costs for Ukrainian: ASCII is one byte per character, Cyrillic is two. Before any
// merges a Ukrainian sentence is twice the sequence length of its English translation, and the
// roadmap's risk R2 — "byte fallback is the primary path, not an edge case" — is about exactly this.
namespace veda::tokenizer
{

// The GPT-2 bijection between the 256 byte values and 256 printable code points.
//
// tokenizer.json is a text file, and the byte 0x20 has no printable spelling — so every byte-level
// tokenizer maps bytes onto printable code points before writing them down:
//
//     printable ASCII and most of Latin-1  ->  itself
//     everything else                      ->  U+0100 + n, in order
//
// which is why a space appears as U+0120 'G-dot' in these files and a Cyrillic word appears as a
// run of Latin-1-looking characters. That is not mojibake to be fixed — it IS the token text, and
// the vocabulary is keyed by it.
std::string bytes_to_token_text(std::string_view bytes);

// The inverse. Throws std::invalid_argument on a code point the mapping does not contain: that
// means the text was not produced by this scheme.
std::string token_text_to_bytes(std::string_view text);

class Vocabulary
{
public:
    // Reads model.vocab from a parsed tokenizer.json.
    static Vocabulary from_tokenizer_json(const io::JsonValue& document);

    // nullopt rather than an exception: a token that is not in the vocabulary is a normal answer
    // during merging, not an error.
    std::optional<int32_t> id_of(const std::string& token_text) const;

    // Throws if the id was never defined — the model produced a token the tokenizer does not know,
    // which is a real bug rather than something to paper over.
    const std::string& text_of(int32_t id) const;
    std::string bytes_of(int32_t id) const;

    bool contains(const std::string& token_text) const;
    size_t size() const noexcept { return by_text_.size(); }

    // No normalisation of any kind — no lowercasing, no NFC, no whitespace trimming. The model was
    // trained on the bytes it was given, and any transformation here changes the ids.
    Vocabulary() = default;

    void add(std::string token_text, int32_t id);

private:
    std::unordered_map<std::string, int32_t> by_text_;
    std::vector<std::string> by_id_;   // indexed by id; ids may leave gaps
};

} // namespace veda::tokenizer

#endif //VEDA_VOCABULARY_H
