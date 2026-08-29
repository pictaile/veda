#ifndef VEDA_TENSOR_H
#define VEDA_TENSOR_H

#include "Shape.h"
#include "Storage.h"
#include "Strides.h"

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <vector>

namespace veda::core
{

// A tensor is a description of a region of memory, not the memory itself:
//
//   storage   WHERE are the numbers?         a shared, reference-counted buffer
//   offset    WHERE does this tensor start?  in elements, from storage[0]
//   shape     HOW BIG is it?
//   strides   HOW do I walk it?             how many elements one step along each axis costs
//
// That separation is what makes reshape / transpose / slice / per-head views free: each builds a
// new description over the same bytes. Several tensors may describe one storage; the storage
// lives until the last of them is gone.
class Tensor
{
public:
    // Allocates its own zero-filled storage. This is how activations are made.
    explicit Tensor(Shape shape);

    // Views an existing storage. This is how weights are handed to layers: no copy, ever.
    // Refuses a view that would run past the end of the storage.
    static Tensor view(std::shared_ptr<Storage> storage, size_t offset, Shape shape);

    // Views an existing storage with strides of the caller's choosing — the general form, used by
    // transpose and slice. Refuses a view that could reach past the end of the storage.
    static Tensor view(std::shared_ptr<Storage> storage, size_t offset, Shape shape,
                       std::vector<size_t> strides);

    const Shape& shape() const noexcept { return shape_; }
    const std::vector<size_t>& strides() const noexcept { return strides_; }
    size_t rank() const noexcept { return shape_.rank(); }
    size_t numel() const noexcept { return shape_.size(); }

    // In elements, from the start of the storage.
    size_t offset() const noexcept { return offset_; }

    const std::shared_ptr<Storage>& storage() const noexcept { return storage_; }

    // Reads one element by its coordinates: offset_ + sum over k of indices[k] * strides_[k].
    // Rank and bounds are asserted in debug builds only (AD6).
    float at(std::initializer_list<size_t> indices) const;

    // True when walking this tensor in index order walks the buffer straight through: no gaps,
    // no jumping backwards. Equivalently: the strides are the ones this shape would have if it
    // had just been allocated.
    bool is_contiguous() const;

    // The same elements read under a different shape, as a view over the same storage. Requires
    // an unchanged element count and a contiguous input; throws otherwise.
    Tensor reshape(const Shape& new_shape) const;

    // The first element of this tensor. The only place where element counting turns into
    // pointer arithmetic — sizeof(float) appears nowhere in this layer.
    float* data() noexcept { return storage_->data() + offset_; }
    const float* data() const noexcept { return storage_->data() + offset_; }

private:
    Tensor(std::shared_ptr<Storage> storage, size_t offset, Shape shape, std::vector<size_t> strides);

    std::shared_ptr<Storage> storage_;
    size_t offset_ = 0;
    Shape shape_;
    std::vector<size_t> strides_;
};

} // namespace veda::core

#endif //VEDA_TENSOR_H
