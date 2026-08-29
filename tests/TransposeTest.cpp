// E1.S2.T6 — transpose()

#include "Shape.h"
#include "Storage.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <memory>
#include <stdexcept>
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
    // transpose twice is the identity — written first, it catches an index mix-up immediately
    {
        const Tensor t = filled(Shape({2, 3}));
        const Tensor back = t.transpose(0, 1).transpose(0, 1);
        CHECK_EQ(back.shape(), t.shape());
        CHECK_EQ(back.strides(), t.strides());
        CHECK_EQ(back.offset(), t.offset());
        CHECK(back.is_contiguous());

        const Tensor u = filled(Shape({2, 3, 4}));
        CHECK_EQ(u.transpose(1, 2).transpose(1, 2).strides(), u.strides());
        CHECK_EQ(u.transpose(0, 2).transpose(0, 2).shape(), u.shape());
    }

    // the worked example:
    //   before  (2,3) strides (3,1)   [[10,20,30],[40,50,60]]
    //   after   (3,2) strides (1,3)   [[10,40],[20,50],[30,60]]
    {
        const Tensor t = filled(Shape({2, 3}));
        const Tensor tt = t.transpose(0, 1);

        CHECK_EQ(tt.shape(), Shape({3, 2}));
        CHECK_EQ(tt.strides(), Strides({1, 3}));

        CHECK_EQ(tt.at({0, 0}), 10.0f);   // 0*1 + 0*3 = 0
        CHECK_EQ(tt.at({0, 1}), 40.0f);   // 0*1 + 1*3 = 3
        CHECK_EQ(tt.at({1, 0}), 20.0f);
        CHECK_EQ(tt.at({1, 1}), 50.0f);
        CHECK_EQ(tt.at({2, 0}), 30.0f);
        CHECK_EQ(tt.at({2, 1}), 60.0f);   // 2*1 + 1*3 = 5

        // the buffer is untouched — the bytes still read 10 20 30 40 50 60
        for (size_t i = 0; i < 6; ++i)
        {
            CHECK_EQ(tt.storage()->data()[i], 10.0f * static_cast<float>(i + 1));
        }

        // and the result is exactly the non-contiguous case T4 predicted
        CHECK(t.is_contiguous());
        CHECK(!tt.is_contiguous());
        CHECK_THROWS_AS(tt.reshape(Shape({6})), std::logic_error);
    }

    // no allocation, no data movement: same storage, same offset, same pointer
    {
        Tensor t = filled(Shape({2, 3}));
        Tensor tt = t.transpose(0, 1);
        CHECK(tt.data() == t.data());
        CHECK(tt.storage() == t.storage());
        CHECK_EQ(tt.offset(), t.offset());
        CHECK_EQ(tt.numel(), t.numel());

        tt.data()[0] = 99.0f;
        CHECK_EQ(t.at({0, 0}), 99.0f);   // shared storage
    }

    // rank 3: only the two named axes move
    {
        const Tensor u = filled(Shape({2, 3, 4}));
        CHECK_EQ(u.strides(), Strides({12, 4, 1}));

        const Tensor swapped = u.transpose(1, 2);
        CHECK_EQ(swapped.shape(), Shape({2, 4, 3}));
        CHECK_EQ(swapped.strides(), Strides({12, 1, 4}));

        // u.at({1, 2, 3}) and swapped.at({1, 3, 2}) are the same element
        CHECK_EQ(swapped.at({1, 3, 2}), u.at({1, 2, 3}));
        CHECK_EQ(swapped.at({0, 0, 0}), u.at({0, 0, 0}));
        CHECK_EQ(swapped.at({1, 0, 2}), u.at({1, 2, 0}));

        // the order of the arguments does not matter
        CHECK_EQ(u.transpose(2, 1).strides(), u.transpose(1, 2).strides());
    }

    // the E7 rearrangement: [B, T, H, Dh] -> [B, H, T, Dh]
    {
        const Tensor q{Shape({1, 8, 1024})};
        const Tensor heads = q.reshape(Shape({1, 8, 16, 64}));   // T5
        const Tensor per_head = heads.transpose(1, 2);           // T6

        CHECK_EQ(per_head.shape(), Shape({1, 16, 8, 64}));
        CHECK_EQ(heads.strides(), Strides({8192, 1024, 64, 1}));
        CHECK_EQ(per_head.strides(), Strides({8192, 64, 1024, 1}));
        CHECK(per_head.data() == q.data());   // nothing was copied along the way

        // and the transpose inside the score computation: K [B,H,T,Dh] -> [B,H,Dh,T]
        const Tensor k_t = per_head.transpose(2, 3);
        CHECK_EQ(k_t.shape(), Shape({1, 16, 64, 8}));
    }

    // transpose works on an already non-contiguous tensor — it never assumes an order
    {
        auto storage = std::make_shared<Storage>(12);
        for (size_t i = 0; i < 12; ++i)
        {
            storage->data()[i] = static_cast<float>(i);
        }
        // every other row of a 4x3 buffer: shape (2,3), strides (6,1)
        const Tensor gapped = Tensor::view(storage, 0, Shape({2, 3}), Strides({6, 1}));
        CHECK(!gapped.is_contiguous());
        CHECK_EQ(gapped.at({1, 2}), 8.0f);

        const Tensor gapped_t = gapped.transpose(0, 1);
        CHECK_EQ(gapped_t.shape(), Shape({3, 2}));
        CHECK_EQ(gapped_t.strides(), Strides({1, 6}));
        CHECK_EQ(gapped_t.at({2, 1}), 8.0f);   // the same element, swapped coordinates
    }

    // a view with an offset keeps it
    {
        auto storage = std::make_shared<Storage>(10);
        for (size_t i = 0; i < 10; ++i)
        {
            storage->data()[i] = static_cast<float>(i);
        }
        const Tensor window = Tensor::view(storage, 4, Shape({2, 3}));
        const Tensor t = window.transpose(0, 1);
        CHECK_EQ(t.offset(), size_t{4});
        CHECK_EQ(t.at({0, 0}), 4.0f);
        CHECK_EQ(t.at({2, 1}), 9.0f);   // 4 + 2*1 + 1*3
    }

    // edge cases
    {
        const Tensor t = filled(Shape({2, 3}));
        const Tensor same = t.transpose(1, 1);        // a no-op is legal
        CHECK_EQ(same.shape(), t.shape());
        CHECK_EQ(same.strides(), t.strides());

        const Tensor vector = filled(Shape({4}));     // rank 1 with (0,0)
        CHECK_EQ(vector.transpose(0, 0).shape(), Shape({4}));
        CHECK_EQ(vector.transpose(0, 0).at({3}), 40.0f);

        // a dimension of size 1 transposes like any other
        const Tensor row = filled(Shape({1, 3}));
        CHECK_EQ(row.transpose(0, 1).shape(), Shape({3, 1}));
        CHECK_EQ(row.transpose(0, 1).at({2, 0}), 30.0f);
    }

#ifndef NDEBUG
    // out-of-range dimensions are loud in debug builds (AD6)
    {
        const Tensor t = filled(Shape({2, 3}));
        CHECK_ABORTS(t.transpose(0, 2));
        CHECK_ABORTS(t.transpose(2, 0));
        CHECK_ABORTS(Tensor{Shape({})}.transpose(0, 0));   // rank 0 has no axes to swap
    }
#endif

    return VEDA_TEST_SUMMARY("TransposeTest");
}
