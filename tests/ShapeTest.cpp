// E1.S1.T1 — Shape

#include "Shape.h"
#include "TestSupport.h"

#include <stdexcept>
#include <vector>

using veda::core::Shape;
using veda::core::to_string;

int main()
{
    // rank and size
    CHECK_EQ(Shape({2, 3}).rank(), size_t{2});
    CHECK_EQ(Shape({2, 3}).size(), size_t{6});
    CHECK_EQ(Shape({2, 3})[1], size_t{3});
    CHECK_EQ(Shape({2, 3, 4}).size(), size_t{24});
    CHECK_EQ(Shape({5}).rank(), size_t{1});
    CHECK_EQ(Shape({5}).size(), size_t{5});

    // rank 0: the empty product is 1, so a scalar is one element
    CHECK_EQ(Shape({}).rank(), size_t{0});
    CHECK_EQ(Shape({}).size(), size_t{1});
    CHECK_EQ(Shape().rank(), size_t{0});
    CHECK_EQ(Shape().size(), size_t{1});

    // a zero dimension makes the tensor empty
    CHECK_EQ(Shape({2, 0, 3}).size(), size_t{0});

    // a real Veda shape: [B=1, T=8, D=64]
    CHECK_EQ(Shape({1, 8, 64}).size(), size_t{512});

    // equality: order matters, rank matters
    CHECK_EQ(Shape({2, 3}), Shape({2, 3}));
    CHECK_NE(Shape({2, 3}), Shape({3, 2}));
    CHECK_NE(Shape({2, 3}), Shape({2, 3, 1}));
    CHECK_EQ(Shape({}), Shape());

    // construction from a vector
    CHECK_EQ(Shape(std::vector<size_t>{2, 3, 4}), Shape({2, 3, 4}));

    // string rendering
    CHECK_EQ(to_string(Shape({2, 3, 4})), std::string("(2, 3, 4)"));
    CHECK_EQ(to_string(Shape({7})), std::string("(7)"));
    CHECK_EQ(to_string(Shape({})), std::string("()"));

    // a dimension product that would wrap around size_t is refused rather than silently truncated
    // — shapes come from weight-file headers, which can be corrupt
    CHECK_THROWS_AS(Shape({size_t{1} << 62, 4}), std::overflow_error);
    CHECK_THROWS_AS(Shape({size_t{1} << 40, size_t{1} << 40}), std::overflow_error);
    CHECK_EQ(Shape({size_t{1} << 62, 2}).size(), size_t{1} << 63);   // still representable
    CHECK_EQ(Shape({size_t{1} << 62, 0, 4}).size(), size_t{0});      // a zero dimension short-circuits

    // indexing past the rank is refused, and the message names the shape
    CHECK_THROWS_AS(Shape({2, 3})[2], std::out_of_range);
    CHECK_THROWS_AS(Shape({})[0], std::out_of_range);
    try
    {
        (void)Shape({2, 3})[5];
    }
    catch (const std::out_of_range& error)
    {
        CHECK(std::string(error.what()).find("(2, 3)") != std::string::npos);
    }

    return VEDA_TEST_SUMMARY("ShapeTest");
}
