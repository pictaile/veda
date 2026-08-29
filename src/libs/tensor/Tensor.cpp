#include "Tensor.h"

#include <cassert>
#include <stdexcept>
#include <utility>

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
    return strides_ == contiguous_strides(shape_);
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
