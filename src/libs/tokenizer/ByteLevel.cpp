#include "ByteLevel.h"

#include "Vocabulary.h"

#include <string>

namespace veda::tokenizer
{

Tokenizer byte_tokenizer()
{
    Vocabulary vocabulary;
    for (int32_t value = 0; value < 256; ++value)
    {
        // Keyed by the printable-code-point spelling, exactly as a real tokenizer.json is, so that
        // encode() and decode() take the same path they take for Qwen3's vocabulary.
        const std::string byte(1, static_cast<char>(static_cast<unsigned char>(value)));
        vocabulary.add(bytes_to_token_text(byte), value);
    }

    return Tokenizer(std::move(vocabulary), Merges{});
}

} // namespace veda::tokenizer
