// E1.S3.T9 — DType

#include "DType.h"
#include "Shape.h"
#include "TestSupport.h"

#include <stdexcept>
#include <string>

using veda::core::byte_length;
using veda::core::DType;
using veda::core::dtype_from_name;
using veda::core::dtype_name;
using veda::core::dtype_size;
using veda::core::Shape;

int main()
{
    // two facts per dtype
    CHECK_EQ(dtype_size(DType::F32), size_t{4});
    CHECK_EQ(dtype_size(DType::BF16), size_t{2});
    CHECK_EQ(std::string(dtype_name(DType::F32)), std::string("F32"));
    CHECK_EQ(std::string(dtype_name(DType::BF16)), std::string("BF16"));

    // name -> enum -> name round trip
    CHECK(dtype_from_name("F32") == DType::F32);
    CHECK(dtype_from_name("BF16") == DType::BF16);
    for (const DType dtype : {DType::F32, DType::BF16})
    {
        CHECK(dtype_from_name(dtype_name(dtype)) == dtype);
    }

    // anything else fails loudly, with the offending name in the message
    CHECK_THROWS_AS(dtype_from_name("F16"), std::invalid_argument);
    CHECK_THROWS_AS(dtype_from_name(""), std::invalid_argument);
    CHECK_THROWS_AS(dtype_from_name("I8"), std::invalid_argument);
    CHECK_THROWS_AS(dtype_from_name("f32"), std::invalid_argument);    // case matters
    CHECK_THROWS_AS(dtype_from_name("BF16 "), std::invalid_argument);
    try
    {
        (void)dtype_from_name("F16");
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message = error.what();
        CHECK(message.find("F16") != std::string::npos);
    }

    // byte extent: element count times element size
    CHECK_EQ(byte_length(Shape({1024}), DType::BF16), size_t{2048});
    CHECK_EQ(byte_length(Shape({1024}), DType::F32), size_t{4096});
    CHECK_EQ(byte_length(Shape({}), DType::F32), size_t{4});          // a scalar is one element
    CHECK_EQ(byte_length(Shape({}), DType::BF16), size_t{2});
    CHECK_EQ(byte_length(Shape({2, 3}), DType::F32), size_t{24});
    CHECK_EQ(byte_length(Shape({0}), DType::F32), size_t{0});         // legal, zero bytes
    CHECK_EQ(byte_length(Shape({2, 0, 3}), DType::BF16), size_t{0});

    // the embedding table of Qwen3-0.6B — the largest tensor in the file, and the LM head too
    CHECK_EQ(byte_length(Shape({151669, 1024}), DType::BF16), size_t{310618112});
    CHECK_EQ(byte_length(Shape({151669, 1024}), DType::F32), size_t{621236224});

    // a header entry, verified the way the loader will verify it:
    //   name "model.layers.0.input_layernorm.weight", dtype "BF16", shape [1024],
    //   offsets [4096, 6144]
    {
        const Shape shape({1024});
        const DType dtype = dtype_from_name("BF16");

        const size_t expected = byte_length(shape, dtype);
        CHECK_EQ(expected, size_t{2048});
        CHECK_EQ(size_t{6144} - size_t{4096}, expected);          // the header agrees

        // the same entry with a corrupt span must not be accepted
        CHECK(size_t{8192} - size_t{4096} != expected);
    }

    // a corrupt header claiming an absurd shape overflows rather than under-allocating.
    // The product of the dimensions is caught by Shape itself; the multiplication by the element
    // size is caught here.
    {
        const size_t huge = size_t{1} << 62;
        CHECK_THROWS_AS(Shape({huge, 4}), std::overflow_error);
        CHECK_THROWS_AS(byte_length(Shape({huge}), DType::F32), std::overflow_error);
        CHECK_EQ(byte_length(Shape({huge}), DType::BF16), huge * 2);   // still representable
    }

    return VEDA_TEST_SUMMARY("DTypeTest");
}
