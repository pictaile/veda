#ifndef VEDA_JSON_H
#define VEDA_JSON_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// A JSON reader for exactly what Veda reads: config.json (a few hundred bytes) and tokenizer.json
// (about 7 MB). Objects, arrays, strings with escapes, numbers, true, false, null — and nothing
// else (AD5).
//
// Every feature a parser supports is a feature that can be wrong, and comments, trailing commas,
// NaN literals and duplicate keys appear in neither file. The deliberate omissions are listed in
// the task document; the important inclusion is \uXXXX with surrogate pairs, because
// tokenizer.json is full of them and E5 depends on them decoding exactly.
//
// veda::io depends on nothing above it — not on nn, not on model. This header does not even need
// core.
namespace veda::io
{

class JsonValue
{
public:
    enum class Type
    {
        Null,
        Bool,
        Number,
        String,
        Array,
        Object,
    };

    JsonValue() = default;

    Type type() const noexcept { return type_; }
    bool is_null() const noexcept { return type_ == Type::Null; }
    bool is_bool() const noexcept { return type_ == Type::Bool; }
    bool is_number() const noexcept { return type_ == Type::Number; }
    bool is_string() const noexcept { return type_ == Type::String; }
    bool is_array() const noexcept { return type_ == Type::Array; }
    bool is_object() const noexcept { return type_ == Type::Object; }

    // Object member by name, array element by index. Both throw std::runtime_error naming what was
    // asked for — a missing key in a config is a broken config, not a default.
    const JsonValue& operator[](const std::string& key) const;
    const JsonValue& operator[](size_t index) const;

    bool contains(const std::string& key) const;

    // Elements of an array, or members of an object.
    size_t size() const noexcept;

    const std::string& as_string() const;
    double as_double() const;
    // Throws if the value is not integral: asking for rms_norm_eps as an int is a caller bug, and
    // truncating it silently would hide one.
    int64_t as_int() const;
    bool as_bool() const;

    // Members in document order. Returned as a vector of pairs rather than a keys() list so that
    // iterating tokenizer.json's 150k-entry vocabulary does not duplicate every key.
    const std::vector<std::pair<std::string, JsonValue>>& members() const;
    std::vector<std::string> keys() const;

    // Builders, used by the parser.
    static JsonValue null_value();
    static JsonValue boolean(bool value);
    static JsonValue number(double value);
    static JsonValue string(std::string value);
    static JsonValue array(std::vector<JsonValue> values);
    static JsonValue object(std::vector<std::pair<std::string, JsonValue>> members);

private:
    Type type_ = Type::Null;
    bool bool_ = false;
    double number_ = 0.0;
    std::string string_;
    std::vector<JsonValue> array_;
    std::vector<std::pair<std::string, JsonValue>> members_;
};

// Parses a complete document. Throws std::runtime_error with a line and column on any failure —
// "parse error" alone is nearly useless on a 7 MB file.
JsonValue parse_json(const std::string& text);

} // namespace veda::io

#endif //VEDA_JSON_H
