#include "Json.h"

#include <charconv>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace veda::io
{

namespace
{
[[noreturn]] void wrong_type(const char* wanted, JsonValue::Type actual)
{
    static const char* names[] = {"null", "bool", "number", "string", "array", "object"};
    throw std::runtime_error(std::string("json: value is ") + names[static_cast<int>(actual)] +
                             ", not " + wanted);
}
} // namespace

JsonValue JsonValue::null_value()
{
    return JsonValue{};
}

JsonValue JsonValue::boolean(bool value)
{
    JsonValue out;
    out.type_ = Type::Bool;
    out.bool_ = value;
    return out;
}

JsonValue JsonValue::number(double value)
{
    JsonValue out;
    out.type_ = Type::Number;
    out.number_ = value;
    return out;
}

JsonValue JsonValue::string(std::string value)
{
    JsonValue out;
    out.type_ = Type::String;
    out.string_ = std::move(value);
    return out;
}

JsonValue JsonValue::array(std::vector<JsonValue> values)
{
    JsonValue out;
    out.type_ = Type::Array;
    out.array_ = std::move(values);
    return out;
}

JsonValue JsonValue::object(std::vector<std::pair<std::string, JsonValue>> members)
{
    JsonValue out;
    out.type_ = Type::Object;
    out.members_ = std::move(members);
    return out;
}

const JsonValue& JsonValue::operator[](const std::string& key) const
{
    if (type_ != Type::Object)
    {
        wrong_type("an object", type_);
    }
    for (const auto& member : members_)
    {
        if (member.first == key)
        {
            return member.second;
        }
    }
    throw std::runtime_error("json: no member named \"" + key + "\"");
}

const JsonValue& JsonValue::operator[](size_t index) const
{
    if (type_ != Type::Array)
    {
        wrong_type("an array", type_);
    }
    if (index >= array_.size())
    {
        throw std::runtime_error("json: index " + std::to_string(index) + " is out of range for an " +
                                 "array of " + std::to_string(array_.size()));
    }
    return array_[index];
}

bool JsonValue::contains(const std::string& key) const
{
    if (type_ != Type::Object)
    {
        return false;
    }
    for (const auto& member : members_)
    {
        if (member.first == key)
        {
            return true;
        }
    }
    return false;
}

size_t JsonValue::size() const noexcept
{
    if (type_ == Type::Array)
    {
        return array_.size();
    }
    if (type_ == Type::Object)
    {
        return members_.size();
    }
    return 0;
}

const std::string& JsonValue::as_string() const
{
    if (type_ != Type::String)
    {
        wrong_type("a string", type_);
    }
    return string_;
}

double JsonValue::as_double() const
{
    if (type_ != Type::Number)
    {
        wrong_type("a number", type_);
    }
    return number_;
}

int64_t JsonValue::as_int() const
{
    if (type_ != Type::Number)
    {
        wrong_type("a number", type_);
    }
    if (number_ != std::trunc(number_) || std::isnan(number_) || std::isinf(number_))
    {
        throw std::runtime_error("json: " + std::to_string(number_) + " is not an integer");
    }
    if (number_ < -9.223372036854775e18 || number_ > 9.223372036854775e18)
    {
        throw std::runtime_error("json: " + std::to_string(number_) + " does not fit in an int64");
    }
    return static_cast<int64_t>(number_);
}

bool JsonValue::as_bool() const
{
    if (type_ != Type::Bool)
    {
        wrong_type("a bool", type_);
    }
    return bool_;
}

const std::vector<std::pair<std::string, JsonValue>>& JsonValue::members() const
{
    if (type_ != Type::Object)
    {
        wrong_type("an object", type_);
    }
    return members_;
}

std::vector<std::string> JsonValue::keys() const
{
    std::vector<std::string> out;
    out.reserve(members().size());
    for (const auto& member : members_)
    {
        out.push_back(member.first);
    }
    return out;
}

namespace
{

// One buffer, one cursor. Nothing here copies the input: substring-per-token is the classic
// accidental quadratic, and tokenizer.json is 7 MB.
class Parser
{
public:
    explicit Parser(const std::string& text) : text_(text) {}

    JsonValue parse_document()
    {
        skip_whitespace();
        if (at_end())
        {
            fail("empty document");
        }
        JsonValue value = parse_value();
        skip_whitespace();
        if (!at_end())
        {
            fail("trailing content after the document");
        }
        return value;
    }

private:
    static constexpr size_t max_depth = 256;

    const std::string& text_;
    size_t position_ = 0;
    size_t depth_ = 0;

    bool at_end() const { return position_ >= text_.size(); }
    char peek() const { return text_[position_]; }

    [[noreturn]] void fail(const std::string& reason) const
    {
        // The position is computed only when throwing, which happens once — so tracking line and
        // column during the parse would be a cost paid on every character for nothing.
        size_t line = 1;
        size_t column = 1;
        for (size_t i = 0; i < position_ && i < text_.size(); ++i)
        {
            if (text_[i] == '\n')
            {
                ++line;
                column = 1;
            }
            else
            {
                ++column;
            }
        }
        throw std::runtime_error("json: " + reason + " at line " + std::to_string(line) +
                                 ", column " + std::to_string(column));
    }

    void skip_whitespace()
    {
        while (!at_end())
        {
            const char c = peek();
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
            {
                ++position_;
            }
            else
            {
                break;
            }
        }
    }

    void expect(char c)
    {
        skip_whitespace();
        if (at_end() || peek() != c)
        {
            fail(std::string("expected '") + c + "'");
        }
        ++position_;
    }

    JsonValue parse_value()
    {
        skip_whitespace();
        if (at_end())
        {
            fail("expected a value");
        }

        switch (peek())
        {
            case '{':
                return parse_object();
            case '[':
                return parse_array();
            case '"':
                return JsonValue::string(parse_string());
            case 't':
                return parse_literal("true", JsonValue::boolean(true));
            case 'f':
                return parse_literal("false", JsonValue::boolean(false));
            case 'n':
                return parse_literal("null", JsonValue::null_value());
            default:
                return parse_number();
        }
    }

    JsonValue parse_literal(const char* literal, JsonValue value)
    {
        const size_t length = std::char_traits<char>::length(literal);
        if (text_.compare(position_, length, literal) != 0)
        {
            fail(std::string("expected '") + literal + "'");
        }
        position_ += length;
        return value;
    }

    // A guard against a hostile file of ten thousand nested arrays: without it the stack, not the
    // parser, decides what happens.
    struct DepthGuard
    {
        Parser& parser;
        explicit DepthGuard(Parser& p) : parser(p)
        {
            if (++parser.depth_ > max_depth)
            {
                parser.fail("nesting deeper than " + std::to_string(max_depth));
            }
        }
        ~DepthGuard() { --parser.depth_; }
    };

    JsonValue parse_object()
    {
        DepthGuard guard(*this);
        expect('{');
        std::vector<std::pair<std::string, JsonValue>> members;

        skip_whitespace();
        if (!at_end() && peek() == '}')
        {
            ++position_;
            return JsonValue::object(std::move(members));
        }

        while (true)
        {
            skip_whitespace();
            if (at_end() || peek() != '"')
            {
                fail("expected a member name");
            }
            std::string key = parse_string();
            expect(':');
            members.emplace_back(std::move(key), parse_value());

            skip_whitespace();
            if (at_end())
            {
                fail("unterminated object");
            }
            if (peek() == ',')
            {
                ++position_;
                continue;
            }
            if (peek() == '}')
            {
                ++position_;
                return JsonValue::object(std::move(members));
            }
            fail("expected ',' or '}'");
        }
    }

    JsonValue parse_array()
    {
        DepthGuard guard(*this);
        expect('[');
        std::vector<JsonValue> values;

        skip_whitespace();
        if (!at_end() && peek() == ']')
        {
            ++position_;
            return JsonValue::array(std::move(values));
        }

        while (true)
        {
            values.push_back(parse_value());

            skip_whitespace();
            if (at_end())
            {
                fail("unterminated array");
            }
            if (peek() == ',')
            {
                ++position_;
                continue;
            }
            if (peek() == ']')
            {
                ++position_;
                return JsonValue::array(std::move(values));
            }
            fail("expected ',' or ']'");
        }
    }

    uint32_t parse_hex4()
    {
        if (position_ + 4 > text_.size())
        {
            fail("truncated \\u escape");
        }
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i)
        {
            const char c = text_[position_++];
            value <<= 4;
            if (c >= '0' && c <= '9')
            {
                value |= static_cast<uint32_t>(c - '0');
            }
            else if (c >= 'a' && c <= 'f')
            {
                value |= static_cast<uint32_t>(c - 'a' + 10);
            }
            else if (c >= 'A' && c <= 'F')
            {
                value |= static_cast<uint32_t>(c - 'A' + 10);
            }
            else
            {
                --position_;
                fail("invalid hex digit in a \\u escape");
            }
        }
        return value;
    }

    static void append_utf8(std::string& out, uint32_t codepoint)
    {
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
    }

    std::string parse_string()
    {
        expect('"');
        std::string out;

        while (true)
        {
            if (at_end())
            {
                fail("unterminated string");
            }
            const char c = text_[position_];

            if (c == '"')
            {
                ++position_;
                return out;
            }
            if (static_cast<unsigned char>(c) < 0x20)
            {
                fail("raw control character in a string");
            }
            if (c != '\\')
            {
                // Raw bytes, including multi-byte UTF-8, pass through untouched.
                out += c;
                ++position_;
                continue;
            }

            ++position_;
            if (at_end())
            {
                fail("unterminated escape");
            }
            const char escape = text_[position_++];
            switch (escape)
            {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u':
                {
                    uint32_t unit = parse_hex4();

                    // A high surrogate means nothing on its own: the character it belongs to is
                    // spread across two \u escapes, and decoding them separately produces two
                    // invalid characters instead of one emoji.
                    if (unit >= 0xD800 && unit <= 0xDBFF)
                    {
                        if (position_ + 1 >= text_.size() || text_[position_] != '\\' ||
                            text_[position_ + 1] != 'u')
                        {
                            fail("high surrogate without a following \\u escape");
                        }
                        position_ += 2;
                        const uint32_t low = parse_hex4();
                        if (low < 0xDC00 || low > 0xDFFF)
                        {
                            fail("high surrogate followed by a non-low surrogate");
                        }
                        unit = 0x10000 + (unit - 0xD800) * 0x400 + (low - 0xDC00);
                    }
                    else if (unit >= 0xDC00 && unit <= 0xDFFF)
                    {
                        fail("lone low surrogate");
                    }
                    append_utf8(out, unit);
                    break;
                }
                default:
                    --position_;
                    fail("unknown escape");
            }
        }
    }

    JsonValue parse_number()
    {
        const size_t start = position_;
        if (!at_end() && (peek() == '-' || peek() == '+'))
        {
            ++position_;
        }
        while (!at_end())
        {
            const char c = peek();
            if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-')
            {
                ++position_;
            }
            else
            {
                break;
            }
        }
        if (start == position_)
        {
            fail("expected a value");
        }

        double value = 0.0;
        const char* first = text_.data() + start;
        const char* last = text_.data() + position_;
        const auto result = std::from_chars(first, last, value);
        if (result.ec != std::errc{} || result.ptr != last)
        {
            position_ = start;
            fail("malformed number");
        }
        return JsonValue::number(value);
    }
};

} // namespace

JsonValue parse_json(const std::string& text)
{
    Parser parser(text);
    return parser.parse_document();
}

} // namespace veda::io
