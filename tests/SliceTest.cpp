// E1.S2.T7 — slice()

#include "Shape.h"
#include "Storage.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <memory>
#include <vector>

using veda::core::Shape;
using veda::core::Storage;
using veda::core::Tensor;
using Strides = std::vector<size_t>;

namespace
{
Tensor filled(const Shape& shape, float first = 10.0f, float step = 10.0f)
{
    Tensor t{shape};
    for (size_t i = 0; i < shape.size(); ++i)
    {
        t.data()[i] = first + step * static_cast<float>(i);
    }
    return t;
}
} // namespace

int main()
{
    // buffer: 10 20 30 40 50 60 read as [[10,20,30],[40,50,60]], strides (3,1)

    // slice along dim 0: one row starting at row 1 -> [[40, 50, 60]]
    {
        const Tensor t = filled(Shape({2, 3}));
        const Tensor row = t.slice(0, 1, 1);

        CHECK_EQ(row.shape(), Shape({1, 3}));
        CHECK_EQ(row.offset(), size_t{3});          // 0 + 1 * strides[0] = 3
        CHECK_EQ(row.strides(), t.strides());       // unchanged
        CHECK_EQ(row.at({0, 0}), 40.0f);
        CHECK_EQ(row.at({0, 2}), 60.0f);
        CHECK(row.is_contiguous());                 // slicing the first dimension preserves it
    }

    // slice along dim 1: two columns starting at column 1 -> [[20,30],[50,60]]
    {
        const Tensor t = filled(Shape({2, 3}));
        const Tensor cols = t.slice(1, 1, 2);

        CHECK_EQ(cols.shape(), Shape({2, 2}));
        CHECK_EQ(cols.offset(), size_t{1});         // 0 + 1 * strides[1] = 1
        CHECK_EQ(cols.strides(), Strides({3, 1}));  // still (3,1), not (2,1)
        CHECK_EQ(cols.at({0, 0}), 20.0f);
        CHECK_EQ(cols.at({0, 1}), 30.0f);
        CHECK_EQ(cols.at({1, 0}), 50.0f);           // offset 1 + 1*3 + 0*1 = 4
        CHECK_EQ(cols.at({1, 1}), 60.0f);

        // not contiguous: contiguous strides for (2,2) would be (2,1). The gap is element 40,
        // which the slice deliberately skips.
        CHECK(!cols.is_contiguous());
    }

    // it is a view: writes are visible in the original
    {
        Tensor t = filled(Shape({2, 3}));
        Tensor s = t.slice(0, 1, 1);
        s.data()[0] = 99.0f;
        CHECK_EQ(t.at({1, 0}), 99.0f);
        CHECK(s.storage() == t.storage());
        CHECK_EQ(t.storage().use_count(), long{2});
    }

    // full slice is equivalent to the original; empty slice is legal
    {
        const Tensor t = filled(Shape({2, 3}));

        const Tensor whole = t.slice(0, 0, 2);
        CHECK_EQ(whole.shape(), t.shape());
        CHECK_EQ(whole.strides(), t.strides());
        CHECK_EQ(whole.offset(), t.offset());

        const Tensor none = t.slice(0, 0, 0);
        CHECK_EQ(none.numel(), size_t{0});
        CHECK_EQ(none.shape(), Shape({0, 3}));

        const Tensor none_inner = t.slice(1, 3, 0);   // empty at the very end of the dimension
        CHECK_EQ(none_inner.numel(), size_t{0});
    }

    // slicing composes with transpose
    {
        const Tensor t = filled(Shape({2, 3}));
        // transposed: (3,2) strides (1,3); slice(0,1,1) -> (1,2) offset 1
        const Tensor s = t.transpose(0, 1).slice(0, 1, 1);
        CHECK_EQ(s.shape(), Shape({1, 2}));
        CHECK_EQ(s.offset(), size_t{1});
        CHECK_EQ(s.strides(), Strides({1, 3}));
        CHECK_EQ(s.at({0, 0}), 20.0f);
        CHECK_EQ(s.at({0, 1}), 50.0f);   // 1 + 0*1 + 1*3 = 4

        // and the other order: slice then transpose
        const Tensor other = t.slice(1, 1, 2).transpose(0, 1);
        CHECK_EQ(other.shape(), Shape({2, 2}));
        CHECK_EQ(other.at({0, 1}), 50.0f);
        CHECK_EQ(other.at({1, 0}), 30.0f);
    }

    // slicing a slice: offsets accumulate
    {
        const Tensor t = filled(Shape({4, 3}));      // 10 .. 120
        const Tensor middle = t.slice(0, 1, 2);      // rows 1..2
        CHECK_EQ(middle.offset(), size_t{3});
        const Tensor inner = middle.slice(1, 2, 1);  // last column of those rows
        CHECK_EQ(inner.offset(), size_t{5});         // 3 + 2*1
        CHECK_EQ(inner.shape(), Shape({2, 1}));
        CHECK_EQ(inner.at({0, 0}), 60.0f);
        CHECK_EQ(inner.at({1, 0}), 90.0f);
    }

    // the generation-loop use: [B, T, V] -> the last position only
    {
        const size_t T = 8;
        const size_t V = 32;                          // 151669 in the real model
        const Tensor logits{Shape({1, T, V})};
        logits.storage()->data()[7 * V + 5] = 3.5f;   // last position, token 5

        const Tensor last = logits.slice(1, T - 1, 1);
        CHECK_EQ(last.shape(), Shape({1, 1, V}));
        CHECK_EQ(last.offset(), size_t{7 * V});       // (T-1) * strides[1] = 7 * V
        CHECK_EQ(last.at({0, 0, 5}), 3.5f);

        // one unbroken run of V floats — so it is contiguous, and the reshape is legal
        CHECK(last.is_contiguous());
        const Tensor flat = last.reshape(Shape({1, V}));
        CHECK_EQ(flat.at({0, 5}), 3.5f);
        CHECK(flat.data() == last.data());            // no copy of 151669 floats per step
    }

    // the KV cache use (E12): allocate once, view the growing valid prefix
    {
        const size_t T_max = 16;
        const Tensor cache{Shape({1, 2, T_max, 4})};   // [B, Hkv, T_max, Dh]
        for (size_t t = 0; t < 3; ++t)
        {
            const Tensor valid = cache.slice(2, 0, t + 1);
            CHECK_EQ(valid.shape(), Shape({1, 2, t + 1, 4}));
            CHECK_EQ(valid.offset(), size_t{0});         // start stays 0
            CHECK_EQ(valid.strides(), cache.strides());  // strides stay fixed
            CHECK(valid.storage() == cache.storage());   // no reallocation
        }
    }

    // slicing the last dimension
    {
        const Tensor t = filled(Shape({2, 3}));
        const Tensor last_col = t.slice(1, 2, 1);
        CHECK_EQ(last_col.shape(), Shape({2, 1}));
        CHECK_EQ(last_col.offset(), size_t{2});
        CHECK_EQ(last_col.at({0, 0}), 30.0f);
        CHECK_EQ(last_col.at({1, 0}), 60.0f);
        CHECK(!last_col.is_contiguous());   // contiguous (2,1) would be (1,1)
    }

    // slicing a rank-1 tensor, and a rank-3 inner dimension
    {
        const Tensor v = filled(Shape({5}));
        const Tensor part = v.slice(0, 2, 2);
        CHECK_EQ(part.shape(), Shape({2}));
        CHECK_EQ(part.at({0}), 30.0f);
        CHECK(part.is_contiguous());

        const Tensor u = filled(Shape({2, 3, 4}));   // strides (12,4,1)
        const Tensor mid = u.slice(1, 1, 1);
        CHECK_EQ(mid.shape(), Shape({2, 1, 4}));
        CHECK_EQ(mid.offset(), size_t{4});
        CHECK_EQ(mid.at({0, 0, 0}), u.at({0, 1, 0}));
        CHECK_EQ(mid.at({1, 0, 3}), u.at({1, 1, 3}));
        CHECK(!mid.is_contiguous());
    }

#ifndef NDEBUG
    // out-of-range slices are loud in debug builds (AD6)
    {
        const Tensor t = filled(Shape({2, 3}));
        CHECK_ABORTS(t.slice(0, 1, 2));   // start + count > 2
        CHECK_ABORTS(t.slice(1, 0, 4));
        CHECK_ABORTS(t.slice(1, 3, 1));
        CHECK_ABORTS(t.slice(2, 0, 1));   // no such dimension
    }
#endif

    return VEDA_TEST_SUMMARY("SliceTest");
}
