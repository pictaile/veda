// E4.S1.T2 — the JSON reader

#include "Json.h"
#include "TestSupport.h"

#include <stdexcept>
#include <string>
#include <vector>

using veda::io::JsonValue;
using veda::io::parse_json;

namespace
{
// The shape of a real config.json, cut down to what Veda reads.
const char* config_text = R"({
  "architectures": ["Qwen3ForCausalLM"],
  "hidden_size": 1024,
  "intermediate_size": 3072,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "num_hidden_layers": 28,
  "head_dim": 64,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "tie_word_embeddings": true,
  "vocab_size": 151936,
  "unused": null
})";

std::string message_of(const std::string& text)
{
    try
    {
        (void)parse_json(text);
    }
    catch (const std::runtime_error& error)
    {
        return error.what();
    }
    return "";
}
} // namespace

int main()
{
    // a real config parses, and every accessor returns what it should
    {
        const JsonValue config = parse_json(config_text);

        CHECK(config.is_object());
        CHECK_EQ(config.size(), size_t{12});
        CHECK_EQ(config["hidden_size"].as_int(), int64_t{1024});
        CHECK_EQ(config["num_hidden_layers"].as_int(), int64_t{28});
        CHECK_EQ(config["num_key_value_heads"].as_int(), int64_t{8});
        CHECK_NEAR(config["rms_norm_eps"].as_double(), 1e-6, 1e-12);
        CHECK_NEAR(config["rope_theta"].as_double(), 1000000.0, 1e-6);
        CHECK(config["tie_word_embeddings"].as_bool());
        CHECK(config["unused"].is_null());
        CHECK_EQ(config["architectures"][0].as_string(), std::string("Qwen3ForCausalLM"));
        CHECK_EQ(config["architectures"].size(), size_t{1});

        CHECK(config.contains("vocab_size"));
        CHECK(!config.contains("rope_scaling"));
        CHECK_EQ(config.keys().front(), std::string("architectures"));   // document order kept
        CHECK_EQ(config.members().size(), size_t{12});
    }

    // the six value types, nested
    {
        const JsonValue value = parse_json(
            R"({"o": {"inner": [1, {"deep": "x"}, [true, false, null]]}, "e1": {}, "e2": []})");

        CHECK(value["o"].is_object());
        CHECK(value["o"]["inner"].is_array());
        CHECK_EQ(value["o"]["inner"].size(), size_t{3});
        CHECK_EQ(value["o"]["inner"][0].as_double(), 1.0);
        CHECK_EQ(value["o"]["inner"][1]["deep"].as_string(), std::string("x"));
        CHECK(value["o"]["inner"][2][0].as_bool());
        CHECK(!value["o"]["inner"][2][1].as_bool());
        CHECK(value["o"]["inner"][2][2].is_null());
        CHECK_EQ(value["e1"].size(), size_t{0});
        CHECK_EQ(value["e2"].size(), size_t{0});
        CHECK(value["e1"].is_object());
        CHECK(value["e2"].is_array());
    }

    // numbers, in every form the two files use
    {
        const JsonValue value =
            parse_json(R"([0, -1, 1024, 1e-06, 1000000.0, -2.5e3, 0.0, 1E3, 3.5])");
        CHECK_EQ(value[0].as_int(), int64_t{0});
        CHECK_EQ(value[1].as_int(), int64_t{-1});
        CHECK_EQ(value[2].as_int(), int64_t{1024});
        CHECK_NEAR(value[3].as_double(), 1e-6, 1e-12);
        CHECK_EQ(value[4].as_int(), int64_t{1000000});     // 1000000.0 is integral
        CHECK_NEAR(value[5].as_double(), -2500.0, 1e-9);
        CHECK_EQ(value[6].as_double(), 0.0);
        CHECK_EQ(value[7].as_int(), int64_t{1000});

        // asking for a non-integral value as an int is a caller bug, and it says so
        CHECK_THROWS_AS(value[3].as_int(), std::runtime_error);
        CHECK_THROWS_AS(value[8].as_int(), std::runtime_error);
        CHECK_THROWS_AS(value[0].as_string(), std::runtime_error);
        CHECK_THROWS_AS(value[0].as_bool(), std::runtime_error);
        CHECK_THROWS_AS(parse_json(R"("text")").as_double(), std::runtime_error);
    }

    // wrong access reports what was asked for
    {
        const JsonValue config = parse_json(config_text);
        CHECK_THROWS_AS(config["nope"], std::runtime_error);
        CHECK_THROWS_AS(config["architectures"][5], std::runtime_error);
        CHECK_THROWS_AS(config["hidden_size"]["x"], std::runtime_error);

        try
        {
            (void)config["nope"];
        }
        catch (const std::runtime_error& error)
        {
            CHECK(std::string(error.what()).find("nope") != std::string::npos);
        }
    }

    // escapes — where the bugs live, and what E5 depends on
    {
        const JsonValue value = parse_json(
            R"(["a\"b", "back\\slash", "line\nbreak", "tab\there", "A", "Ж",
                "中", "😀", "\/slash", "\b\f\r"])");

        CHECK_EQ(value[0].as_string(), std::string("a\"b"));
        CHECK_EQ(value[1].as_string(), std::string("back\\slash"));
        CHECK_EQ(value[2].as_string(), std::string("line\nbreak"));
        CHECK_EQ(value[3].as_string(), std::string("tab\there"));

        CHECK_EQ(value[4].as_string(), std::string("A"));            // one byte
        CHECK_EQ(value[5].as_string(), std::string("Ж"));            // two bytes, Cyrillic
        CHECK_EQ(value[5].as_string().size(), size_t{2});
        CHECK_EQ(value[6].as_string(), std::string("中"));           // three bytes
        CHECK_EQ(value[6].as_string().size(), size_t{3});
        CHECK_EQ(value[7].as_string(), std::string("😀"));           // four, from a surrogate pair
        CHECK_EQ(value[7].as_string().size(), size_t{4});
        CHECK_EQ(value[8].as_string(), std::string("/slash"));
        CHECK_EQ(value[9].as_string().size(), size_t{3});

        // raw multi-byte UTF-8 passes through untouched
        CHECK_EQ(parse_json("[\"сон\"]")[0].as_string(), std::string("сон"));
        CHECK_EQ(parse_json("[\" сон\"]")[0].as_string().size(), size_t{7});   // leading space

        // a lone surrogate is rejected rather than silently producing invalid UTF-8
        CHECK_THROWS_AS(parse_json(R"(["\ud83d"])"), std::runtime_error);
        CHECK_THROWS_AS(parse_json(R"(["\ud83dx"])"), std::runtime_error);
        CHECK_THROWS_AS(parse_json(R"(["\ude00"])"), std::runtime_error);
        CHECK_THROWS_AS(parse_json(R"(["\ud83dA"])"), std::runtime_error);
        CHECK_THROWS_AS(parse_json(R"(["\u00zz"])"), std::runtime_error);
        CHECK_THROWS_AS(parse_json(R"(["\q"])"), std::runtime_error);
    }

    // malformed input: every failure carries a line and a column
    {
        for (const char* text : {R"({"a": })", R"({"a" 1})", R"({"a": 1,})", R"({"a": 1)",
                                 "[1, 2", R"("unterminated)", "{", "[", "", "   ",
                                 R"({"a": 1} extra)", "tru", R"({a: 1})", R"([1 2])"})
        {
            CHECK_THROWS_AS(parse_json(text), std::runtime_error);
        }

        CHECK(message_of(R"({"a": })").find("line 1") != std::string::npos);
        CHECK(message_of(R"({"a": })").find("column") != std::string::npos);
        CHECK(message_of(R"({"a" 1})").find("expected ':'") != std::string::npos);
        CHECK(message_of(R"({"a": 1} extra)").find("trailing content") != std::string::npos);
        CHECK(message_of("").find("empty document") != std::string::npos);

        // the line number is real, not always 1
        const std::string multiline = "{\n  \"a\": 1,\n  \"b\": ,\n}";
        CHECK(message_of(multiline).find("line 3") != std::string::npos);
    }

    // control characters inside strings are not allowed raw
    {
        CHECK_THROWS_AS(parse_json("[\"a\nb\"]"), std::runtime_error);
        CHECK_EQ(parse_json("[\"a\\nb\"]")[0].as_string(), std::string("a\nb"));
    }

    // top-level scalars, and whitespace tolerance
    {
        CHECK_EQ(parse_json("  42  ").as_int(), int64_t{42});
        CHECK(parse_json("\n\ttrue\r\n").as_bool());
        CHECK(parse_json(" null ").is_null());
        CHECK_EQ(parse_json(R"( "x" )").as_string(), std::string("x"));
    }

    // a nesting cap, so a hostile file cannot decide what the stack does
    {
        std::string deep;
        for (int i = 0; i < 300; ++i)
        {
            deep += "[";
        }
        deep += "1";
        for (int i = 0; i < 300; ++i)
        {
            deep += "]";
        }
        CHECK_THROWS_AS(parse_json(deep), std::runtime_error);
        CHECK(message_of(deep).find("nesting deeper") != std::string::npos);

        // just under the cap is fine
        std::string shallow;
        for (int i = 0; i < 100; ++i)
        {
            shallow += "[";
        }
        shallow += "7";
        for (int i = 0; i < 100; ++i)
        {
            shallow += "]";
        }
        const JsonValue nested = parse_json(shallow);
        CHECK(nested.is_array());
    }

    // scale: the shape of tokenizer.json, at a size that would expose an accidental quadratic
    {
        std::string big = R"({"model": {"vocab": {)";
        const size_t entries = 50000;
        for (size_t i = 0; i < entries; ++i)
        {
            if (i > 0)
            {
                big += ",";
            }
            big += "\"tok" + std::to_string(i) + "\": " + std::to_string(i);
        }
        big += "}, \"merges\": [\"\\u0120 t\", \"h e\"]}}";

        const JsonValue parsed = parse_json(big);
        CHECK_EQ(parsed["model"]["vocab"].size(), entries);
        CHECK_EQ(parsed["model"]["vocab"]["tok0"].as_int(), int64_t{0});
        CHECK_EQ(parsed["model"]["merges"].size(), size_t{2});
        CHECK_EQ(parsed["model"]["merges"][0].as_string(), std::string("Ġ t"));

        // and iterating members costs one pass, without duplicating 50k keys
        CHECK_EQ(parsed["model"]["vocab"].members().back().first, std::string("tok49999"));
        CHECK_EQ(parsed["model"]["vocab"].members().back().second.as_int(), int64_t{49999});
    }

    return VEDA_TEST_SUMMARY("JsonTest");
}
