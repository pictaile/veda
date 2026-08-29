// E1.S2.T5 — is_contiguous() and reshape()

#include "Shape.h"
#include "Storage.h"
#include "Strides.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using veda::core::Shape;
using veda::core::Storage;
using veda::core::Tensor;
using Strides = std::vector<size_t>;

namespace
{
// buffer: 10 20 30 40 50 60, read as (2, 3)
Tensor filled_2x3()
{
    Tensor t{Shape({2, 3})};
    for (size_t i = 0; i < 6; ++i)
    {
        t.data()[i] = static_cast<float>(10 * (i + 1));
    }
    return t;
}
} // namespace

int main()
{
    // is_contiguous
    {
        CHECK(Tensor{Shape({2, 3})}.is_contiguous());
        CHECK(Tensor{Shape({2, 3, 4})}.is_contiguous());
        CHECK(Tensor{Shape({})}.is_contiguous());          // a scalar, by definition
        CHECK(Tensor{Shape({1})}.is_contiguous());
        CHECK(Tensor{Shape({2, 0, 3})}.is_contiguous());   // empty, still contiguous

        auto storage = std::make_shared<Storage>(12);
        CHECK(Tensor::view(storage, 3, Shape({3, 2})).is_contiguous());

        // transposed: strides swapped, shape (3,2) would want (2,1)
        CHECK(!Tensor::view(storage, 0, Shape({3, 2}), Strides({1, 3})).is_contiguous());
        // gapped: every other row of a 4x3 buffer
        CHECK(!Tensor::view(storage, 0, Shape({2, 3}), Strides({6, 1})).is_contiguous());
    }

    // reshape (2,3) -> (6,): same buffer, fresh strides (1,)
    {
        const Tensor t = filled_2x3();
        const Tensor flat = t.reshape(Shape({6}));

        CHECK_EQ(flat.shape(), Shape({6}));
        CHECK_EQ(flat.strides(), Strides({1}));
        CHECK_EQ(flat.at({0}), 10.0f);
        CHECK_EQ(flat.at({4}), 50.0f);
        CHECK_EQ(flat.at({5}), 60.0f);
    }

    // reshape (2,3) -> (3,2): reads as [[10,20],[30,40],[50,60]]
    {
        const Tensor t = filled_2x3();
        const Tensor r = t.reshape(Shape({3, 2}));

        CHECK_EQ(r.strides(), Strides({2, 1}));   // fresh contiguous strides, not the old ones
        CHECK_EQ(r.at({0, 0}), 10.0f);
        CHECK_EQ(r.at({1, 1}), 40.0f);
        CHECK_EQ(r.at({2, 0}), 50.0f);
        CHECK(r.is_contiguous());
    }

    // identity reshape is legal
    {
        const Tensor t = filled_2x3();
        const Tensor same = t.reshape(Shape({2, 3}));
        CHECK_EQ(same.shape(), Shape({2, 3}));
        CHECK_EQ(same.at({1, 2}), 60.0f);
    }

    // the head split of E7: [B, T, D] -> [B, T, H, Dh], not one float copied
    {
        const Tensor q{Shape({1, 8, 1024})};
        const Tensor heads = q.reshape(Shape({1, 8, 16, 64}));
        CHECK_EQ(heads.shape(), Shape({1, 8, 16, 64}));
        CHECK_EQ(heads.strides(), Strides({8192, 1024, 64, 1}));
        CHECK(heads.data() == q.data());               // same memory
        CHECK(heads.storage() == q.storage());
    }

    // it is a view, not a copy
    {
        Tensor t = filled_2x3();
        Tensor r = t.reshape(Shape({6}));
        r.data()[0] = 99.0f;
        CHECK_EQ(t.at({0, 0}), 99.0f);
        CHECK_EQ(t.storage().use_count(), long{2});    // shared storage, no allocation
        CHECK_EQ(r.offset(), t.offset());
    }

    // a reshape of a view keeps the view's offset
    {
        auto storage = std::make_shared<Storage>(10);
        for (size_t i = 0; i < 10; ++i)
        {
            storage->data()[i] = static_cast<float>(i);
        }
        const Tensor window = Tensor::view(storage, 4, Shape({2, 3}));
        const Tensor r = window.reshape(Shape({3, 2}));
        CHECK_EQ(r.offset(), size_t{4});
        CHECK_EQ(r.at({0, 0}), 4.0f);
        CHECK_EQ(r.at({2, 1}), 9.0f);
    }

    // refusals: element count, and contiguity — with distinguishable messages
    {
        const Tensor t = filled_2x3();
        CHECK_THROWS_AS(t.reshape(Shape({4})), std::invalid_argument);
        CHECK_THROWS_AS(t.reshape(Shape({2, 4})), std::invalid_argument);
        CHECK_THROWS_AS(t.reshape(Shape({})), std::invalid_argument);

        try
        {
            (void)t.reshape(Shape({4}));
        }
        catch (const std::exception& error)
        {
            const std::string message = error.what();
            CHECK(message.find("(2, 3) -> (4)") != std::string::npos);
            CHECK(message.find("element count 6 != 4") != std::string::npos);
        }

        // a transposed view has the right element count and must still be refused:
        // reading straight through would give 10 20 30 40 50 60 instead of 10 40 20 50 30 60
        auto storage = t.storage();
        const Tensor transposed = Tensor::view(storage, 0, Shape({3, 2}), Strides({1, 3}));
        CHECK_EQ(transposed.at({0, 1}), 40.0f);   // the view itself is correct
        CHECK_THROWS_AS(transposed.reshape(Shape({6})), std::logic_error);

        try
        {
            (void)transposed.reshape(Shape({6}));
        }
        catch (const std::exception& error)
        {
            const std::string message = error.what();
            CHECK(message.find("not contiguous") != std::string::npos);
            CHECK(message.find("(1, 3)") != std::string::npos);   // the offending strides
        }
    }

    // edge cases
    {
        Tensor one{Shape({1, 1})};
        one.data()[0] = 7.0f;
        const Tensor scalar = one.reshape(Shape({}));      // rank 0
        CHECK_EQ(scalar.shape().rank(), size_t{0});
        CHECK_EQ(scalar.strides(), Strides({}));
        CHECK_EQ(scalar.at({}), 7.0f);
        CHECK_EQ(scalar.reshape(Shape({1})).at({0}), 7.0f); // and back

        const Tensor empty{Shape({2, 0, 3})};              // a zero dimension
        CHECK_EQ(empty.reshape(Shape({0})).shape(), Shape({0}));
        CHECK_EQ(empty.reshape(Shape({0, 5})).numel(), size_t{0});
    }

    return VEDA_TEST_SUMMARY("ReshapeTest");
}
