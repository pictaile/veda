// E1.S1.T2 — Storage and Tensor

#include "Shape.h"
#include "Storage.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>

using veda::core::Shape;
using veda::core::Storage;
using veda::core::Tensor;

int main()
{
    // Storage owns a zero-filled flat buffer and cannot be copied
    static_assert(!std::is_copy_constructible_v<Storage>);
    static_assert(!std::is_copy_assignable_v<Storage>);
    {
        const Storage storage(6);
        CHECK_EQ(storage.size(), size_t{6});
        CHECK_EQ(storage.data()[0], 0.0f);
        CHECK_EQ(storage.data()[5], 0.0f);
    }

    // A tensor built from a shape allocates its own storage
    {
        Tensor t{Shape({2, 3})};
        CHECK_EQ(t.shape(), Shape({2, 3}));
        CHECK_EQ(t.rank(), size_t{2});
        CHECK_EQ(t.numel(), size_t{6});
        CHECK_EQ(t.offset(), size_t{0});
        CHECK_EQ(t.storage()->size(), size_t{6});
        CHECK_EQ(t.data()[0], 0.0f);
    }

    // Two views over one storage: the write through B is visible through A.
    // Storage:  10 20 30 40 50 60
    // A: offset 0, (2,3)   B: offset 3, (3)
    {
        auto storage = std::make_shared<Storage>(6);
        for (size_t i = 0; i < 6; ++i)
        {
            storage->data()[i] = static_cast<float>(10 * (i + 1));
        }

        Tensor a = Tensor::view(storage, 0, Shape({2, 3}));
        Tensor b = Tensor::view(storage, 3, Shape({3}));

        CHECK_EQ(a.numel(), size_t{6});
        CHECK_EQ(b.numel(), size_t{3});
        CHECK_EQ(b.offset(), size_t{3});
        CHECK_EQ(b.data()[0], 40.0f);   // b starts at element 3 of the storage

        b.data()[0] = 99.0f;
        CHECK_EQ(a.data()[3], 99.0f);   // a view is not a snapshot
        CHECK_EQ(storage->data()[3], 99.0f);

        // no copy happened: both tensors point into the one buffer
        CHECK(a.data() == storage->data());
        CHECK(b.data() == storage->data() + 3);
    }

    // Reference counting: the storage outlives whoever created it
    {
        auto storage = std::make_shared<Storage>(4);
        CHECK_EQ(storage.use_count(), long{1});

        std::optional<Tensor> a = Tensor::view(storage, 0, Shape({4}));
        CHECK_EQ(storage.use_count(), long{2});

        std::optional<Tensor> b = Tensor::view(storage, 2, Shape({2}));
        CHECK_EQ(storage.use_count(), long{3});

        a.reset();
        CHECK_EQ(storage.use_count(), long{2});   // b keeps it alive

        b->data()[0] = 7.0f;
        CHECK_EQ(storage->data()[2], 7.0f);
        b.reset();
        CHECK_EQ(storage.use_count(), long{1});
    }

    // A tensor that outlives the shared_ptr the caller held keeps the buffer alive
    {
        std::optional<Tensor> survivor;
        {
            auto storage = std::make_shared<Storage>(3);
            storage->data()[1] = 5.0f;
            survivor = Tensor::view(storage, 0, Shape({3}));
        }
        CHECK_EQ(survivor->data()[1], 5.0f);
        CHECK_EQ(survivor->storage().use_count(), long{1});
    }

    // Edge cases
    {
        // a zero-element tensor is legal and allocates nothing to describe
        Tensor empty{Shape({2, 0, 3})};
        CHECK_EQ(empty.numel(), size_t{0});
        CHECK_EQ(empty.storage()->size(), size_t{0});

        // a rank-0 tensor is one element
        Tensor scalar{Shape({})};
        CHECK_EQ(scalar.numel(), size_t{1});
        CHECK_EQ(scalar.storage()->size(), size_t{1});

        auto storage = std::make_shared<Storage>(6);

        // a view at the very end, describing nothing, is legal
        Tensor tail = Tensor::view(storage, 6, Shape({0}));
        CHECK_EQ(tail.numel(), size_t{0});

        // exactly filling the storage is legal
        CHECK_EQ(Tensor::view(storage, 3, Shape({3})).numel(), size_t{3});

        // running past the end is refused
        CHECK_THROWS_AS(Tensor::view(storage, 4, Shape({3})), std::out_of_range);
        CHECK_THROWS_AS(Tensor::view(storage, 7, Shape({0})), std::out_of_range);
        CHECK_THROWS_AS(Tensor::view(nullptr, 0, Shape({1})), std::invalid_argument);
    }

    return VEDA_TEST_SUMMARY("TensorTest");
}
