#ifndef VEDA_TOKENIZER_H
#define VEDA_TOKENIZER_H

#include "Json.h"
#include "Vocabulary.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace veda::tokenizer
{

// The ordered list of merges learned during training. Position in the list is the rank: the first
// entry was the most frequent pair in the corpus, and encoding replays that order.
class Merges
{
public:
    // Reads model.merges, which may be an array of "left right" strings or of two-element arrays.
    static Merges from_tokenizer_json(const io::JsonValue& document);

    std::optional<size_t> rank(const std::string& left, const std::string& right) const;
    size_t size() const noexcept { return ranks_.size(); }

    void add(const std::string& left, const std::string& right, size_t rank);

private:
    std::unordered_map<std::string, size_t> ranks_;   // keyed by left + '\x01' + right
};

// Splits text into the pieces merging happens within — merges never cross a piece boundary, or
// common cross-word pairs would merge and the vocabulary would fill with nonsense spanning spaces.
//
// The reference rule is a Unicode-property regex; std::regex has no \p{L}, so this implements the
// same structure with a byte-level classification: an ASCII letter or any byte >= 0x80 counts as a
// letter, ASCII digits as digits. That treats emoji and symbols as letters where the reference
// treats them as "other" — for Latin or Cyrillic prose the two agree, and only S3's corpus
// comparison against the reference tokenizer can prove it in general. The limitation is written
// down so that a later mismatch has a place to start.
//
// The pieces concatenate back to the input exactly: no trimming, no substitution, no normalisation.
std::vector<std::string> pre_tokenize(std::string_view text);

class Tokenizer
{
public:
    Tokenizer(Vocabulary vocabulary, Merges merges);
    static Tokenizer from_tokenizer_json(const io::JsonValue& document);

    // Splits into pieces, then merges within each: repeatedly take the adjacent pair with the
    // lowest merge rank and merge every occurrence of it, until no pair has a merge.
    //
    // This is not longest-match. Longest-match asks "what is the biggest vocabulary entry that fits
    // here"; BPE asks "what would training have merged first", and the two disagree whenever an
    // early merge cuts across a longer entry. Both give plausible tokens; only one gives the ids the
    // model was trained on (risk R2).
    std::vector<int32_t> encode(std::string_view text) const;

    // Concatenates each id's bytes. A special id decodes to its literal content, never through the
    // byte mapping. For a complete sequence the result is always valid UTF-8, because the sequence
    // came from valid UTF-8 — the incremental case is what StreamingDecoder exists for.
    std::string decode(const std::vector<int32_t>& ids) const;

    // Special tokens are stored as literal text, not in byte-level form, and bypass BPE entirely:
    // merging "<|endoftext|>" would produce a run of ordinary tokens spelling the same characters
    // and meaning something else to the model.
    void add_special_token(std::string content, int32_t id);
    const std::vector<std::pair<std::string, int32_t>>& special_tokens() const noexcept
    {
        return specials_;
    }

    const Vocabulary& vocabulary() const noexcept { return vocabulary_; }
    const Merges& merges() const noexcept { return merges_; }

private:
    // The bytes an id stands for, or its literal content if it is special.
    std::string bytes_of_id(int32_t id) const;
    std::vector<int32_t> encode_ordinary(std::string_view text) const;

    Vocabulary vocabulary_;
    Merges merges_;
    std::vector<std::pair<std::string, int32_t>> specials_;   // longest content first
    std::unordered_map<int32_t, std::string> special_by_id_;
};

// Generation emits one token at a time, and a token is a run of bytes rather than characters: a
// Cyrillic letter is two bytes, an emoji four, and nothing stops a merge from ending between them.
// So a single token can decode to half a character, and printing that half produces an invalid byte
// sequence.
//
// This holds back an incomplete tail — at most three bytes, since that is the longest possible
// truncation of a UTF-8 character — and emits only what is complete.
class StreamingDecoder
{
public:
    explicit StreamingDecoder(const Tokenizer& tokenizer);

    std::string push(int32_t id);

    // Whatever remains once the sequence is over. A partial character here means the model stopped
    // mid-character; returning the bytes raw is the honest answer, and a caller that wants U+FFFD
    // can add one.
    std::string flush();

    size_t held() const noexcept { return buffer_.size(); }

private:
    const Tokenizer& tokenizer_;
    std::string buffer_;
};

} // namespace veda::tokenizer

#endif //VEDA_TOKENIZER_H
