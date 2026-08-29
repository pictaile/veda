// E1.S2.T4 — Contiguity (a Learn task; is_contiguous() itself arrives in T5)
//
// Walks a shape in index order — last dimension fastest — and records the buffer offset each
// step lands on. Contiguity is exactly the question of whether that walk goes straight through.

#include "Shape.h"
#include "Strides.h"
#include "TestSupport.h"

#include <vector>

using veda::core::contiguous_strides;
using veda::core::Shape;
using Offsets = std::vector<size_t>;
using Strides = std::vector<size_t>;

// The offsets visited walking the tensor in index order, using the offset formula from T3.
Offsets walk(const Shape& shape, const Strides& strides)
{
    Offsets offsets;
    offsets.reserve(shape.size());

    std::vector<size_t> index(shape.rank(), 0);
    for (size_t step = 0; step < shape.size(); ++step)
    {
        size_t flat = 0;
        for (size_t k = 0; k < shape.rank(); ++k)
        {
            flat += index[k] * strides[k];
        }
        offsets.push_back(flat);

        // advance the last dimension fastest, carrying leftwards
        for (size_t k = shape.rank(); k-- > 0;)
        {
            if (++index[k] < shape[k])
            {
                break;
            }
            index[k] = 0;
        }
    }
    return offsets;
}

bool is_contiguous(const Shape& shape, const Strides& strides)
{
    return strides == contiguous_strides(shape);
}

int main()
{
    // buffer:  10 20 30 40 50 60
    //
    // contiguous 2x3, strides (3, 1): the walk goes straight through
    CHECK_EQ(walk(Shape({2, 3}), Strides({3, 1})), Offsets({0, 1, 2, 3, 4, 5}));

    // the same buffer transposed to 3x2, strides (1, 3): the walk jumps around
    CHECK_EQ(walk(Shape({3, 2}), Strides({1, 3})), Offsets({0, 3, 1, 4, 2, 5}));

    // which is why reshape of the transposed view to (6,) would read the buffer straight through
    // — 10 20 30 40 50 60 — while the tensor logically holds 10 40 20 50 30 60.

    CHECK(is_contiguous(Shape({2, 3}), Strides({3, 1})));
    CHECK(!is_contiguous(Shape({3, 2}), Strides({1, 3})));   // transposed
    CHECK(is_contiguous(Shape({6}), Strides({1})));
    CHECK(!is_contiguous(Shape({2, 3}), Strides({6, 1})));   // gaps: every other row
    CHECK(is_contiguous(Shape({1, 8, 64}), Strides({512, 64, 1})));

    // a gapped view: every other row of a 4x3 buffer
    CHECK_EQ(walk(Shape({2, 3}), Strides({6, 1})), Offsets({0, 1, 2, 6, 7, 8}));

    // right shape, wrong strides — shape alone never tells you
    CHECK(!is_contiguous(Shape({2, 3}), Strides({1, 2})));

    // last stride 1 is necessary but not sufficient
    CHECK_EQ(contiguous_strides(Shape({2, 3})).back(), size_t{1});
    CHECK(!is_contiguous(Shape({2, 3}), Strides({6, 1})));   // last stride is 1, still gapped

    // a freshly computed stride set is contiguous for every shape, by definition
    for (const Shape& shape : {Shape({}), Shape({5}), Shape({2, 3}), Shape({2, 3, 4})})
    {
        CHECK(is_contiguous(shape, contiguous_strides(shape)));
    }

    return VEDA_TEST_SUMMARY("ContiguityTest");
}
