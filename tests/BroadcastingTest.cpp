// E2.S1.T1 — Broadcasting (a Learn task; ops::broadcast_shape / broadcast_to arrive in T2)
//
// Works out the rule on paper, in code: how two shapes combine, and how a stride of zero turns
// repetition into a view instead of a copy.

#include "Shape.h"
#include "Storage.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <memory>
#include <optional>
#include <vector>

using veda::core::Shape;
using veda::core::Storage;
using veda::core::Tensor;
using Strides = std::vector<size_t>;

namespace
{
// The rule: align from the right; each pair must be equal, or one of them must be 1. A missing
// dimension on the left counts as 1. Returns nothing when the shapes do not broadcast.
std::optional<Shape> broadcast_shape(const Shape& a, const Shape& b)
{
    const size_t rank = a.rank() > b.rank() ? a.rank() : b.rank();
    std::vector<size_t> dims(rank);

    for (size_t i = 0; i < rank; ++i)
    {
        // walk from the right: position i counted from the end of each shape
        const size_t da = i < a.rank() ? a[a.rank() - 1 - i] : 1;
        const size_t db = i < b.rank() ? b[b.rank() - 1 - i] : 1;

        if (da != db && da != 1 && db != 1)
        {
            return std::nullopt;
        }
        dims[rank - 1 - i] = da == 1 ? db : da;   // 0 against 1 stays 0, so not simply the larger
    }
    return Shape(std::move(dims));
}

// Strides that read `shape` as if it were `target`: zero wherever an axis of extent 1 is repeated,
// and zero for the leading axes the input does not have at all.
Strides broadcast_strides(const Shape& shape, const Strides& strides, const Shape& target)
{
    Strides out(target.rank(), 0);
    for (size_t i = 0; i < shape.rank(); ++i)
    {
        const size_t from_end = shape.rank() - 1 - i;
        out[target.rank() - 1 - i] = shape[from_end] == 1 ? 0 : strides[from_end];
    }
    return out;
}
} // namespace

int main()
{
    // a zero extent against 1 stays zero: the size-1 side repeats zero times
    CHECK_EQ(*broadcast_shape(Shape({0}), Shape({})), Shape({0}));
    CHECK_EQ(*broadcast_shape(Shape({0, 3}), Shape({1, 3})), Shape({0, 3}));
    CHECK_EQ(*broadcast_shape(Shape({1}), Shape({0})), Shape({0}));

    // the rule, case by case
    CHECK_EQ(*broadcast_shape(Shape({2, 3}), Shape({2, 3})), Shape({2, 3}));   // equal
    CHECK_EQ(*broadcast_shape(Shape({2, 3}), Shape({2, 1})), Shape({2, 3}));   // one side is 1
    CHECK_EQ(*broadcast_shape(Shape({2, 1}), Shape({2, 3})), Shape({2, 3}));   // symmetric
    CHECK_EQ(*broadcast_shape(Shape({2, 3}), Shape({3})), Shape({2, 3}));      // missing axis
    CHECK_EQ(*broadcast_shape(Shape({1, 8, 64}), Shape({8, 1})), Shape({1, 8, 64}));
    CHECK_EQ(*broadcast_shape(Shape({}), Shape({2, 3})), Shape({2, 3}));       // a scalar fits all

    // the three real uses in the forward pass
    CHECK_EQ(*broadcast_shape(Shape({1, 8, 64}), Shape({64})), Shape({1, 8, 64}));        // bias
    CHECK_EQ(*broadcast_shape(Shape({1, 8, 64}), Shape({1, 8, 1})), Shape({1, 8, 64}));   // scale
    CHECK_EQ(*broadcast_shape(Shape({1, 16, 8, 8}), Shape({8, 8})), Shape({1, 16, 8, 8})); // mask

    // and the refusals
    CHECK(!broadcast_shape(Shape({2, 3}), Shape({3, 2})).has_value());
    CHECK(!broadcast_shape(Shape({2, 3}), Shape({4})).has_value());
    CHECK(!broadcast_shape(Shape({8, 4}), Shape({2, 4, 4})).has_value());

    // a stride of zero is the mechanism: advancing that axis skips nothing, so every index along
    // it reads the same element
    {
        auto storage = std::make_shared<Storage>(3);
        storage->data()[0] = 7.0f;
        storage->data()[1] = 8.0f;
        storage->data()[2] = 9.0f;

        const Tensor bias = Tensor::view(storage, 0, Shape({3}));       // strides (1)
        const Shape target({2, 3});
        const Strides strides = broadcast_strides(bias.shape(), bias.strides(), target);
        CHECK_EQ(strides, Strides({0, 1}));

        const Tensor spread = Tensor::view(storage, 0, target, strides);
        CHECK_EQ(spread.shape(), Shape({2, 3}));

        // both rows read the same three numbers, from the same three floats
        CHECK_EQ(spread.at({0, 0}), 7.0f);
        CHECK_EQ(spread.at({1, 0}), 7.0f);
        CHECK_EQ(spread.at({0, 1}), spread.at({1, 1}));
        CHECK_EQ(spread.at({0, 2}), spread.at({1, 2}));

        // no copy: the storage is still three floats, held by the two views and this handle
        CHECK_EQ(spread.storage()->size(), size_t{3});
        CHECK(spread.storage() == bias.storage());

        // a broadcast view is not contiguous — which is why ops must walk it through its strides
        CHECK(!spread.is_contiguous());
    }

    // a per-token scale [T, 1] against [T, D]: the zero goes on the last axis instead
    {
        auto storage = std::make_shared<Storage>(2);
        storage->data()[0] = 10.0f;
        storage->data()[1] = 20.0f;

        const Tensor scale = Tensor::view(storage, 0, Shape({2, 1}));   // strides (1, 1)
        const Shape target({2, 3});
        const Strides strides = broadcast_strides(scale.shape(), scale.strides(), target);
        CHECK_EQ(strides, Strides({1, 0}));

        const Tensor spread = Tensor::view(storage, 0, target, strides);
        CHECK_EQ(spread.at({0, 0}), 10.0f);
        CHECK_EQ(spread.at({0, 2}), 10.0f);   // the whole first row is one number
        CHECK_EQ(spread.at({1, 0}), 20.0f);
        CHECK_EQ(spread.at({1, 2}), 20.0f);

        // x * scale, worked out by hand: [[1,2,3],[4,5,6]] * [[10],[20]]
        Tensor x{Shape({2, 3})};
        for (size_t i = 0; i < 6; ++i)
        {
            x.data()[i] = static_cast<float>(i + 1);
        }
        CHECK_EQ(x.at({1, 2}) * spread.at({1, 2}), 120.0f);
        CHECK_EQ(x.at({0, 0}) * spread.at({0, 0}), 10.0f);
    }

    // what broadcasting saves, in the shape that matters most: a causal mask at H = 16, T = 1024
    {
        const Shape mask({1024, 1024});
        const Shape scores({1, 16, 1024, 1024});
        CHECK_EQ(*broadcast_shape(scores, mask), scores);

        const size_t materialised = scores.size();   // what a copy would cost, per layer
        const size_t broadcast = mask.size();        // what the view costs
        CHECK_EQ(materialised, size_t{16777216});    // 16M floats, 64 MB
        CHECK_EQ(broadcast, size_t{1048576});
        CHECK_EQ(materialised / broadcast, size_t{16});   // sixteen identical copies of one triangle
    }

    return VEDA_TEST_SUMMARY("BroadcastingTest");
}
