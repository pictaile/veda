#include "Broadcast.h"

#include <stdexcept>
#include <vector>

namespace veda::ops
{

using core::Shape;
using core::Tensor;

Shape broadcast_shape(const Shape& a, const Shape& b)
{
    const size_t rank = a.rank() > b.rank() ? a.rank() : b.rank();
    std::vector<size_t> dims(rank);

    for (size_t i = 0; i < rank; ++i)
    {
        // position i counted from the right of each shape; a missing dimension counts as 1
        const size_t extent_a = i < a.rank() ? a[a.rank() - 1 - i] : 1;
        const size_t extent_b = i < b.rank() ? b[b.rank() - 1 - i] : 1;

        if (extent_a != extent_b && extent_a != 1 && extent_b != 1)
        {
            throw std::invalid_argument(
                "broadcast_shape: " + a.to_string() + " and " + b.to_string() +
                " do not broadcast: " + std::to_string(extent_a) + " against " +
                std::to_string(extent_b) + " at dimension " + std::to_string(rank - 1 - i));
        }
        // The side that is 1 gives way to the other — which is not the same as taking the
        // larger: an extent of 0 against 1 must stay 0, because the size-1 side is repeated zero
        // times. Taking the maximum would claim an element that does not exist.
        dims[rank - 1 - i] = extent_a == 1 ? extent_b : extent_a;
    }
    return Shape(std::move(dims));
}

Tensor broadcast_to(const Tensor& tensor, const Shape& target)
{
    const Shape& shape = tensor.shape();

    if (shape.rank() > target.rank())
    {
        throw std::invalid_argument(
            "broadcast_to: " + shape.to_string() + " has a higher rank than the target " +
            target.to_string());
    }

    // Leading axes the input does not have are repeated wholesale, so they start at zero.
    std::vector<size_t> strides(target.rank(), 0);

    for (size_t i = 0; i < shape.rank(); ++i)
    {
        const size_t from_end = shape.rank() - 1 - i;
        const size_t extent = shape[from_end];
        const size_t wanted = target[target.rank() - 1 - i];

        if (extent != wanted && extent != 1)
        {
            throw std::invalid_argument(
                "broadcast_to: " + shape.to_string() + " cannot be read as " + target.to_string() +
                ": " + std::to_string(extent) + " against " + std::to_string(wanted) +
                " at dimension " + std::to_string(target.rank() - 1 - i));
        }

        // extent 1 against a wider target repeats: stride 0. Otherwise the axis keeps its own step.
        strides[target.rank() - 1 - i] = extent == 1 ? 0 : tensor.strides()[from_end];
    }

    return Tensor::view(tensor.storage(), tensor.offset(), target, std::move(strides));
}

} // namespace veda::ops
