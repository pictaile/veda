#ifndef VEDA_SHAPE_H
#define VEDA_SHAPE_H

#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <string>
#include <vector>

namespace veda::core
{
    // The size of a tensor: one extent per dimension.
    //
    // A deliberately distinct type rather than an alias for std::vector<size_t>, so that a stride
    // vector can never be passed where a shape is expected. Immutable after construction: a shape
    // that changes under a tensor that was built from it is a whole class of bug.
    class Shape
    {
    public:
        // Rank-0: a scalar. Its size is 1 — the empty product.
        Shape() = default;

        explicit Shape(std::vector<size_t> dims);
        Shape(std::initializer_list<size_t> dims);

        // Number of dimensions.
        size_t rank() const noexcept { return dims_.size(); }

        // Total number of elements: the product of all dimensions, 1 for rank 0.
        size_t size() const noexcept { return size_; }

        // Extent of dimension i. Throws std::out_of_range if i >= rank().
        size_t operator[](size_t i) const;

        const std::vector<size_t>& dims() const noexcept { return dims_; }

        bool operator==(const Shape& other) const noexcept { return dims_ == other.dims_; }
        bool operator!=(const Shape& other) const noexcept { return !(*this == other); }

        // Renders as "(2, 3, 4)"; a rank-0 shape renders as "()".
        std::string to_string() const;

    private:
        std::vector<size_t> dims_;
        size_t size_ = 1;
    };

    std::string to_string(const Shape& shape);
    std::ostream& operator<<(std::ostream& os, const Shape& shape);
} // namespace veda::core

#endif //VEDA_SHAPE_H
