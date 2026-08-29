#include "Tensor.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <utility>
#include <vector>

namespace veda::core
{

Tensor::Tensor(Shape shape)
    : storage_(std::make_shared<Storage>(shape.size())),
      offset_(0),
      shape_(std::move(shape)),
      strides_(contiguous_strides(shape_))
{
}

Tensor::Tensor(std::shared_ptr<Storage> storage, size_t offset, Shape shape, std::vector<size_t> strides)
    : storage_(std::move(storage)), offset_(offset), shape_(std::move(shape)), strides_(std::move(strides))
{
}

Tensor Tensor::view(std::shared_ptr<Storage> storage, size_t offset, Shape shape)
{
    std::vector<size_t> strides = contiguous_strides(shape);
    return view(std::move(storage), offset, std::move(shape), std::move(strides));
}

Tensor Tensor::view(std::shared_ptr<Storage> storage, size_t offset, Shape shape,
                    std::vector<size_t> strides)
{
    if (!storage)
    {
        throw std::invalid_argument("Tensor::view: storage is null");
    }
    if (shape.rank() != strides.size())
    {
        throw std::invalid_argument(
            "Tensor::view: shape " + shape.to_string() + " has rank " + std::to_string(shape.rank()) +
            " but " + std::to_string(strides.size()) + " strides were given");
    }

    const size_t span = span_in_elements(shape, strides);
    if (offset > storage->size() || span > storage->size() - offset)
    {
        throw std::out_of_range(
            "Tensor::view: shape " + shape.to_string() + " with strides " + to_string(strides) +
            " at offset " + std::to_string(offset) + " runs past the end of a storage of " +
            std::to_string(storage->size()) + " elements");
    }

    return Tensor(std::move(storage), offset, std::move(shape), std::move(strides));
}

bool Tensor::is_contiguous() const
{
    if (shape_.size() == 0)
    {
        return true; // nothing to walk
    }

    // Walking in index order must walk the buffer straight through. Checked from the right, each
    // axis must step over the whole block of axes to its right.
    //
    // Axes of extent 1 are skipped: their index is always 0, so they are never stepped along and
    // their stride constrains nothing. Slicing [B, T, V] down to the last position leaves
    // strides (T*V, V, 1) under shape (1, 1, V) — still one unbroken run, and comparing against
    // contiguous_strides() alone would wrongly call it strided.
    size_t expected = 1;
    for (size_t k = shape_.rank(); k-- > 0;)
    {
        const size_t extent = shape_[k];
        if (extent == 1)
        {
            continue;
        }
        if (strides_[k] != expected)
        {
            return false;
        }
        expected *= extent;
    }
    return true;
}

Tensor Tensor::reshape(const Shape& new_shape) const
{
    if (new_shape.size() != shape_.size())
    {
        throw std::invalid_argument(
            "Tensor::reshape " + shape_.to_string() + " -> " + new_shape.to_string() +
            ": element count " + std::to_string(shape_.size()) + " != " + std::to_string(new_shape.size()));
    }
    if (!is_contiguous())
    {
        throw std::logic_error(
            "Tensor::reshape " + shape_.to_string() + " -> " + new_shape.to_string() +
            ": tensor is not contiguous (strides " + to_string(strides_) + ", contiguous would be " +
            to_string(contiguous_strides(shape_)) + "); make a contiguous copy first");
    }

    // Same storage, same offset, same bytes — only the reading rule changes.
    return Tensor(storage_, offset_, new_shape, contiguous_strides(new_shape));
}

Tensor Tensor::transpose(size_t dim_a, size_t dim_b) const
{
    assert(dim_a < shape_.rank() && "Tensor::transpose: first dimension is out of range");
    assert(dim_b < shape_.rank() && "Tensor::transpose: second dimension is out of range");

    // A stride is "how far to step to advance this axis by one", so swapping the axes means
    // swapping their strides. The offset formula then reaches the same buffer position from the
    // swapped coordinates: i*s0 + j*s1 == j*s1 + i*s0.
    std::vector<size_t> dims = shape_.dims();
    std::vector<size_t> strides = strides_;
    std::swap(dims[dim_a], dims[dim_b]);
    std::swap(strides[dim_a], strides[dim_b]);

    return Tensor(storage_, offset_, Shape(std::move(dims)), std::move(strides));
}

Tensor Tensor::slice(size_t dim, size_t start, size_t count) const
{
    assert(dim < shape_.rank() && "Tensor::slice: dimension is out of range");
    assert(start + count <= shape_[dim] && "Tensor::slice: range is out of bounds");

    // Skipping the first `start` entries along a dimension means moving the starting point
    // forward by `start` steps of that dimension's stride. Within the slice, advancing any
    // dimension still costs what it did before — so the strides are carried over untouched.
    const size_t new_offset = offset_ + start * strides_[dim];

    std::vector<size_t> dims = shape_.dims();
    dims[dim] = count;

    return Tensor(storage_, new_offset, Shape(std::move(dims)), strides_);
}

float Tensor::at(std::initializer_list<size_t> indices) const
{
    assert(indices.size() == shape_.rank() && "Tensor::at: wrong number of indices");

    size_t flat = offset_;
    size_t k = 0;
    for (const size_t index : indices)
    {
        assert(index < shape_[k] && "Tensor::at: index out of bounds");
        flat += index * strides_[k];
        ++k;
    }

    assert(flat < storage_->size() && "Tensor::at: computed offset is outside the storage");
    return storage_->data()[flat];
}

} // namespace veda::core
