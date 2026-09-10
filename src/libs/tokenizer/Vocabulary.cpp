#include "Vocabulary.h"

#include <array>
#include <stdexcept>

namespace veda::tokenizer
{

namespace
{

struct ByteMapping
{
    std::array<std::string, 256> to_text;                    // byte -> its code point, as UTF-8
    std::unordered_map<std::string, unsigned char> to_byte;  // and back

    ByteMapping()
    {
        // The rule: printable ASCII and most of Latin-1 map to themselves; every other byte maps to
        // U+0100 + n, in order of appearance.
        std::array<bool, 256> printable{};
        for (int b = '!'; b <= '~'; ++b)
        {
            printable[static_cast<size_t>(b)] = true;
        }
        for (int b = 0xA1; b <= 0xAC; ++b)
        {
            printable[static_cast<size_t>(b)] = true;
        }
        for (int b = 0xAE; b <= 0xFF; ++b)
        {
            printable[static_cast<size_t>(b)] = true;
        }

        int shifted = 0;
        for (int b = 0; b < 256; ++b)
        {
            const uint32_t codepoint =
                printable[static_cast<size_t>(b)] ? static_cast<uint32_t>(b)
                                                  : static_cast<uint32_t>(0x100 + shifted++);

            std::string encoded;
            if (codepoint < 0x80)
            {
                encoded += static_cast<char>(codepoint);
            }
            else if (codepoint < 0x800)
            {
                encoded += static_cast<char>(0xC0 | (codepoint >> 6));
                encoded += static_cast<char>(0x80 | (codepoint & 0x3F));
            }
            else
            {
                encoded += static_cast<char>(0xE0 | (codepoint >> 12));
                encoded += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                encoded += static_cast<char>(0x80 | (codepoint & 0x3F));
            }

            to_text[static_cast<size_t>(b)] = encoded;
            to_byte[encoded] = static_cast<unsigned char>(b);
        }
    }
};

const ByteMapping& mapping()
{
    // Built once: the rule is fiddly enough that recomputing it per call would be both slow and a
    // second place to get it wrong.
    static const ByteMapping table;
    return table;
}

size_t utf8_length(unsigned char lead)
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
    return 4;
}

} // namespace

std::string bytes_to_token_text(std::string_view bytes)
{
    std::string out;
    out.reserve(bytes.size() * 2);
    for (const char byte : bytes)
    {
        out += mapping().to_text[static_cast<unsigned char>(byte)];
    }
    return out;
}

std::string token_text_to_bytes(std::string_view text)
{
    std::string out;
    out.reserve(text.size());

    size_t position = 0;
    while (position < text.size())
    {
        const size_t width = utf8_length(static_cast<unsigned char>(text[position]));
        if (position + width > text.size())
        {
            throw std::invalid_argument("tokenizer: truncated UTF-8 in token text");
        }

        const std::string character(text.substr(position, width));
        const auto found = mapping().to_byte.find(character);
        if (found == mapping().to_byte.end())
        {
            throw std::invalid_argument("tokenizer: \"" + character +
                                        "\" is not part of the byte-level mapping");
        }
        out += static_cast<char>(found->second);
        position += width;
    }
    return out;
}

void Vocabulary::add(std::string token_text, int32_t id)
{
    if (id < 0)
    {
        throw std::invalid_argument("tokenizer: negative token id " + std::to_string(id));
    }
    const size_t index = static_cast<size_t>(id);
    if (by_id_.size() <= index)
    {
        by_id_.resize(index + 1);
    }
    by_id_[index] = token_text;
    by_text_.emplace(std::move(token_text), id);
}

Vocabulary Vocabulary::from_tokenizer_json(const io::JsonValue& document)
{
    if (!document.is_object() || !document.contains("model"))
    {
        throw std::runtime_error("tokenizer: the document has no \"model\" object");
    }
    const io::JsonValue& model = document["model"];
    if (!model.contains("vocab"))
    {
        throw std::runtime_error("tokenizer: the model has no \"vocab\" object");
    }

    Vocabulary vocabulary;
    // members() rather than keys(): iterating a 150k-entry vocabulary should not duplicate every
    // key (E4.S1.T2).
    for (const auto& member : model["vocab"].members())
    {
        vocabulary.add(member.first, static_cast<int32_t>(member.second.as_int()));
    }
    return vocabulary;
}

std::optional<int32_t> Vocabulary::id_of(const std::string& token_text) const
{
    const auto found = by_text_.find(token_text);
    if (found == by_text_.end())
    {
        return std::nullopt;
    }
    return found->second;
}

const std::string& Vocabulary::text_of(int32_t id) const
{
    const size_t index = static_cast<size_t>(id);
    if (id < 0 || index >= by_id_.size() || by_id_[index].empty())
    {
        throw std::out_of_range("tokenizer: no token with id " + std::to_string(id));
    }
    return by_id_[index];
}

std::string Vocabulary::bytes_of(int32_t id) const
{
    return token_text_to_bytes(text_of(id));
}

bool Vocabulary::contains(const std::string& token_text) const
{
    return by_text_.contains(token_text);
}

} // namespace veda::tokenizer
