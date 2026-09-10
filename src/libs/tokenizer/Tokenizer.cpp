#include "Tokenizer.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace veda::tokenizer
{

namespace
{

std::string pair_key(const std::string& left, const std::string& right)
{
    return left + '\x01' + right;
}

bool is_ascii_letter(unsigned char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

bool is_letter(unsigned char c)
{
    // Any byte >= 0x80 is part of a multi-byte character, and for this purpose those count as
    // letters — which is what makes Cyrillic and CJK behave like words rather than punctuation.
    return is_ascii_letter(c) || c >= 0x80;
}

bool is_digit(unsigned char c)
{
    return c >= '0' && c <= '9';
}

bool is_space(unsigned char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

// UTF-8 says how long a character is in its lead byte, so a truncated tail is recognisable without
// any table: 0xxxxxxx is one byte, 110xxxxx two, 1110xxxx three, 11110xxx four.
size_t utf8_width(unsigned char lead)
{
    if ((lead & 0x80) == 0)
    {
        return 1;
    }
    if ((lead & 0xE0) == 0xC0)
    {
        return 2;
    }
    if ((lead & 0xF0) == 0xE0)
    {
        return 3;
    }
    if ((lead & 0xF8) == 0xF0)
    {
        return 4;
    }
    return 1;   // a stray continuation byte: emit it rather than hanging on it forever
}

// The length of the longest prefix of `bytes` that is complete UTF-8.
size_t complete_prefix(const std::string& bytes)
{
    size_t position = 0;
    while (position < bytes.size())
    {
        const size_t width = utf8_width(static_cast<unsigned char>(bytes[position]));
        if (position + width > bytes.size())
        {
            break;   // the character is cut off — hold it
        }
        position += width;
    }
    return position;
}

} // namespace

void Merges::add(const std::string& left, const std::string& right, size_t rank)
{
    ranks_.emplace(pair_key(left, right), rank);
}

Merges Merges::from_tokenizer_json(const io::JsonValue& document)
{
    if (!document.is_object() || !document.contains("model"))
    {
        throw std::runtime_error("tokenizer: the document has no \"model\" object");
    }
    const io::JsonValue& model = document["model"];
    if (!model.contains("merges"))
    {
        throw std::runtime_error("tokenizer: the model has no \"merges\" array");
    }

    const io::JsonValue& list = model["merges"];
    if (!list.is_array())
    {
        throw std::runtime_error("tokenizer: \"merges\" is not an array");
    }

    Merges merges;
    for (size_t rank = 0; rank < list.size(); ++rank)
    {
        const io::JsonValue& entry = list[rank];

        if (entry.is_array())
        {
            if (entry.size() != 2)
            {
                throw std::runtime_error("tokenizer: merge " + std::to_string(rank) +
                                         " is not a pair");
            }
            merges.add(entry[0].as_string(), entry[1].as_string(), rank);
            continue;
        }

        const std::string& text = entry.as_string();
        const size_t space = text.find(' ');
        if (space == std::string::npos)
        {
            throw std::runtime_error("tokenizer: merge " + std::to_string(rank) + " \"" + text +
                                     "\" has no separator");
        }
        merges.add(text.substr(0, space), text.substr(space + 1), rank);
    }
    return merges;
}

std::optional<size_t> Merges::rank(const std::string& left, const std::string& right) const
{
    const auto found = ranks_.find(pair_key(left, right));
    if (found == ranks_.end())
    {
        return std::nullopt;
    }
    return found->second;
}

std::vector<std::string> pre_tokenize(std::string_view text)
{
    std::vector<std::string> pieces;
    size_t position = 0;

    while (position < text.size())
    {
        const size_t start = position;
        const unsigned char first = static_cast<unsigned char>(text[position]);

        // Contractions are their own piece: 's 't 're 've 'm 'll 'd
        if (first == '\'' && position + 1 < text.size())
        {
            static const char* contractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
            bool matched = false;
            for (const char* contraction : contractions)
            {
                const std::string_view candidate(contraction);
                if (text.compare(position, candidate.size(), candidate) == 0)
                {
                    pieces.emplace_back(text.substr(position, candidate.size()));
                    position += candidate.size();
                    matched = true;
                    break;
                }
            }
            if (matched)
            {
                continue;
            }
        }

        // An optional leading non-letter (in practice the space), then a run of letters — which is
        // how the leading space comes to belong to the following word.
        size_t look = position;
        if (!is_letter(first) && !is_digit(first) && first != '\r' && first != '\n' &&
            look + 1 < text.size() && is_letter(static_cast<unsigned char>(text[look + 1])))
        {
            ++look;
        }
        if (look < text.size() && is_letter(static_cast<unsigned char>(text[look])))
        {
            ++look;
            while (look < text.size() && is_letter(static_cast<unsigned char>(text[look])))
            {
                ++look;
            }
            pieces.emplace_back(text.substr(start, look - start));
            position = look;
            continue;
        }

        // Digits, one at a time: a number never becomes a single token.
        if (is_digit(first))
        {
            pieces.emplace_back(text.substr(position, 1));
            ++position;
            continue;
        }

        // Newline runs, possibly preceded by other whitespace.
        if (is_space(first))
        {
            size_t whitespace = position;
            while (whitespace < text.size() && is_space(static_cast<unsigned char>(text[whitespace])))
            {
                ++whitespace;
            }
            // A whitespace run that ends the text, or is followed by a non-space, stays one piece.
            pieces.emplace_back(text.substr(position, whitespace - position));
            position = whitespace;
            continue;
        }

        // An optional space, then a run of punctuation.
        size_t punctuation = position;
        while (punctuation < text.size())
        {
            const unsigned char c = static_cast<unsigned char>(text[punctuation]);
            if (is_letter(c) || is_digit(c) || is_space(c))
            {
                break;
            }
            ++punctuation;
        }
        if (punctuation == position)
        {
            ++punctuation;
        }
        pieces.emplace_back(text.substr(position, punctuation - position));
        position = punctuation;
    }

    return pieces;
}

Tokenizer::Tokenizer(Vocabulary vocabulary, Merges merges)
    : vocabulary_(std::move(vocabulary)), merges_(std::move(merges))
{
}

Tokenizer Tokenizer::from_tokenizer_json(const io::JsonValue& document)
{
    Tokenizer tokenizer(Vocabulary::from_tokenizer_json(document),
                        Merges::from_tokenizer_json(document));

    if (document.contains("added_tokens"))
    {
        const io::JsonValue& added = document["added_tokens"];
        for (size_t i = 0; i < added.size(); ++i)
        {
            const io::JsonValue& entry = added[i];
            if (entry.contains("content") && entry.contains("id"))
            {
                tokenizer.add_special_token(entry["content"].as_string(),
                                            static_cast<int32_t>(entry["id"].as_int()));
            }
        }
    }
    return tokenizer;
}

StreamingDecoder::StreamingDecoder(const Tokenizer& tokenizer) : tokenizer_(tokenizer) {}

std::string StreamingDecoder::push(int32_t id)
{
    buffer_ += tokenizer_.decode({id});

    const size_t complete = complete_prefix(buffer_);
    std::string out = buffer_.substr(0, complete);
    buffer_.erase(0, complete);
    return out;
}

std::string StreamingDecoder::flush()
{
    std::string out = std::move(buffer_);
    buffer_.clear();
    return out;
}

void Tokenizer::add_special_token(std::string content, int32_t id)
{
    special_by_id_.emplace(id, content);
    specials_.emplace_back(std::move(content), id);

    // Longest first, so that a special which is a prefix of another does not win.
    std::sort(specials_.begin(), specials_.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
}

std::string Tokenizer::bytes_of_id(int32_t id) const
{
    const auto special = special_by_id_.find(id);
    if (special != special_by_id_.end())
    {
        return special->second;
    }
    return vocabulary_.bytes_of(id);
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const
{
    std::string out;
    for (const int32_t id : ids)
    {
        out += bytes_of_id(id);
    }
    return out;
}

std::vector<int32_t> Tokenizer::encode(std::string_view text) const
{
    // Special tokens are found before anything else and split the text around them: each match
    // becomes its id directly, and the pieces between go through pre-tokenization and merging.
    if (!specials_.empty())
    {
        for (size_t position = 0; position < text.size(); ++position)
        {
            for (const auto& [content, id] : specials_)
            {
                if (!content.empty() && text.compare(position, content.size(), content) == 0)
                {
                    std::vector<int32_t> ids = encode(text.substr(0, position));
                    ids.push_back(id);
                    const std::vector<int32_t> rest =
                        encode(text.substr(position + content.size()));
                    ids.insert(ids.end(), rest.begin(), rest.end());
                    return ids;
                }
            }
        }
    }

    return encode_ordinary(text);
}

std::vector<int32_t> Tokenizer::encode_ordinary(std::string_view text) const
{
    std::vector<int32_t> ids;

    for (const std::string& piece : pre_tokenize(text))
    {
        // One symbol per byte, in the byte-level text form the vocabulary is keyed by.
        std::vector<std::string> symbols;
        symbols.reserve(piece.size());
        for (const char byte : piece)
        {
            symbols.push_back(bytes_to_token_text(std::string_view(&byte, 1)));
        }

        while (symbols.size() > 1)
        {
            size_t best_rank = std::numeric_limits<size_t>::max();
            size_t best_index = 0;
            bool found = false;

            for (size_t i = 0; i + 1 < symbols.size(); ++i)
            {
                const std::optional<size_t> rank = merges_.rank(symbols[i], symbols[i + 1]);
                if (rank && *rank < best_rank)
                {
                    best_rank = *rank;
                    best_index = i;
                    found = true;
                }
            }

            if (!found)
            {
                break;
            }

            // Merge EVERY occurrence of the winning pair before re-scanning: that is what the
            // reference implementations do, and doing one at a time can pick a different second
            // merge.
            const std::string left = symbols[best_index];
            const std::string right = symbols[best_index + 1];

            std::vector<std::string> merged;
            merged.reserve(symbols.size());
            for (size_t i = 0; i < symbols.size();)
            {
                if (i + 1 < symbols.size() && symbols[i] == left && symbols[i + 1] == right)
                {
                    merged.push_back(left + right);
                    i += 2;
                }
                else
                {
                    merged.push_back(symbols[i]);
                    ++i;
                }
            }
            symbols = std::move(merged);
        }

        for (const std::string& symbol : symbols)
        {
            const std::optional<int32_t> id = vocabulary_.id_of(symbol);
            if (!id)
            {
                // The 256 byte tokens are always present, so this can only mean the merge list and
                // the vocabulary disagree — a broken tokenizer.json.
                throw std::runtime_error("tokenizer: \"" + symbol + "\" is not in the vocabulary");
            }
            ids.push_back(*id);
        }
    }

    return ids;
}

} // namespace veda::tokenizer
