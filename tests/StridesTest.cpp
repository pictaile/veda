// E1.S1.T3 — Strides and at()

#include "Shape.h"
#include "Storage.h"
#include "Strides.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <memory>
#include <vector>

using veda::core::contiguous_strides;
using veda::core::Shape;
using veda::core::span_in_elements;
using veda::core::Storage;
using veda::core::Tensor;
using Strides = std::vector<size_t>;

int main()
{
    // shape -> strides, by hand:
    //   (2, 3):     s1 = 1,  s0 = d1 * s1 = 3
    //   (2, 3, 4):  s2 = 1,  s1 = 4,  s0 = d1 * s1 = 12   (one 3x4 slab)
    CHECK_EQ(contiguous_strides(Shape({2, 3})), Strides({3, 1}));
    CHECK_EQ(contiguous_strides(Shape({2, 3, 4})), Strides({12, 4, 1}));
    CHECK_EQ(contiguous_strides(Shape({5})), Strides({1}));
    CHECK_EQ(contiguous_strides(Shape({})), Strides({}));

    // a real Veda shape: [B=1, T=8, D=64] — one token position is 64 floats apart
    CHECK_EQ(contiguous_strides(Shape({1, 8, 64})), Strides({512, 64, 1}));

    // a dimension of size 1 is not special-cased
    CHECK_EQ(contiguous_strides(Shape({1, 1, 3})), Strides({3, 3, 1}));
    CHECK_EQ(contiguous_strides(Shape({4, 1})), Strides({1, 1}));

    // the last stride of a contiguous tensor is always 1
    for (const Shape& shape : {Shape({6}), Shape({2, 3}), Shape({2, 3, 4}), Shape({1, 8, 64})})
    {
        CHECK_EQ(contiguous_strides(shape).back(), size_t{1});
    }

    // span: how far a tensor reaches from its own start
    CHECK_EQ(span_in_elements(Shape({2, 3}), Strides({3, 1})), size_t{6});
    CHECK_EQ(span_in_elements(Shape({}), Strides({})), size_t{1});
    CHECK_EQ(span_in_elements(Shape({2, 0}), Strides({0, 1})), size_t{0});
    // a transposed 2x3 view (T6) is not contiguous but still spans 6
    CHECK_EQ(span_in_elements(Shape({3, 2}), Strides({1, 3})), size_t{6});
    // every second element of a 6-buffer: 3 elements, but reaching across 5
    CHECK_EQ(span_in_elements(Shape({3}), Strides({2})), size_t{5});

    // Tensors carry contiguous strides from construction
    {
        Tensor t{Shape({2, 3, 4})};
        CHECK_EQ(t.strides(), Strides({12, 4, 1}));

        auto storage = std::make_shared<Storage>(6);
        CHECK_EQ(Tensor::view(storage, 3, Shape({3})).strides(), Strides({1}));
    }

    // at() against a buffer verified by hand:
    //   values  [[10, 20, 30],
    //            [40, 50, 60]]
    //   buffer   10 20 30 40 50 60      strides (3, 1)
    {
        Tensor t{Shape({2, 3})};
        for (size_t i = 0; i < 6; ++i)
        {
            t.data()[i] = static_cast<float>(10 * (i + 1));
        }

        CHECK_EQ(t.at({0, 0}), 10.0f);   // 0*3 + 0*1 = 0
        CHECK_EQ(t.at({0, 1}), 20.0f);   // 0*3 + 1*1 = 1
        CHECK_EQ(t.at({0, 2}), 30.0f);
        CHECK_EQ(t.at({1, 0}), 40.0f);   // 1*3 + 0*1 = 3
        CHECK_EQ(t.at({1, 1}), 50.0f);
        CHECK_EQ(t.at({1, 2}), 60.0f);   // 1*3 + 2*1 = 5
    }

    // rank 1 uses the same formula, no special case
    {
        Tensor t{Shape({4})};
        for (size_t i = 0; i < 4; ++i)
        {
            t.data()[i] = static_cast<float>(i);
        }
        CHECK_EQ(t.at({0}), 0.0f);
        CHECK_EQ(t.at({3}), 3.0f);
    }

    // rank 3, and a dimension of size 1
    {
        Tensor t{Shape({2, 3, 4})};
        for (size_t i = 0; i < 24; ++i)
        {
            t.data()[i] = static_cast<float>(i);
        }
        CHECK_EQ(t.at({0, 0, 0}), 0.0f);
        CHECK_EQ(t.at({0, 1, 2}), 6.0f);    // 1*4 + 2*1
        CHECK_EQ(t.at({1, 0, 0}), 12.0f);   // one whole 3x4 slab
        CHECK_EQ(t.at({1, 2, 3}), 23.0f);   // 12 + 8 + 3

        Tensor single{Shape({1, 3})};
        single.data()[2] = 8.0f;
        CHECK_EQ(single.at({0, 2}), 8.0f);
    }

    // at() includes the tensor's own offset: a view starting mid-buffer indexes from its own zero
    {
        auto storage = std::make_shared<Storage>(6);
        for (size_t i = 0; i < 6; ++i)
        {
            storage->data()[i] = static_cast<float>(10 * (i + 1));
        }
        Tensor second_row = Tensor::view(storage, 3, Shape({3}));
        CHECK_EQ(second_row.at({0}), 40.0f);
        CHECK_EQ(second_row.at({2}), 60.0f);

        Tensor as_matrix = Tensor::view(storage, 2, Shape({2, 2}));
        CHECK_EQ(as_matrix.at({0, 0}), 30.0f);
        CHECK_EQ(as_matrix.at({1, 1}), 60.0f);   // 2 + 1*2 + 1*1 = 5
    }

#ifndef NDEBUG
    // Debug builds are loud about shape mistakes (AD6)
    {
        Tensor t{Shape({2, 3})};
        CHECK_ABORTS(t.at({1}));          // too few indices
        CHECK_ABORTS(t.at({1, 1, 1}));    // too many
        CHECK_ABORTS(t.at({2, 0}));       // first index out of bounds
        CHECK_ABORTS(t.at({0, 3}));       // second index out of bounds
    }
#endif

    return VEDA_TEST_SUMMARY("StridesTest");
}
