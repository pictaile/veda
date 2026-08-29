#include "Shape.h"

#include <functional>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace veda::core
{
    namespace
    {
        // std::accumulate with an initial value of 1 gives the empty product for free:
        // the size of a rank-0 shape is 1, not 0.
        size_t element_count(const std::vector<size_t>& dims)
        {
            return std::accumulate(
                dims.begin(),
                dims.end(),
                size_t{1},
                std::multiplies<size_t>()
                );
        }
    } // namespace

    Shape::Shape(std::vector<size_t> dims)
        : dims_(std::move(dims)), size_(element_count(dims_))
    {
    }

    Shape::Shape(std::initializer_list<size_t> dims)
        : dims_(dims), size_(element_count(dims_))
    {
    }

    size_t Shape::operator[](size_t i) const
    {
        if (i >= dims_.size())
        {
            throw std::out_of_range(
                "Shape: dimension index " + std::to_string(i) + " is out of range for shape " + to_string());
        }
        return dims_[i];
    }

    std::string Shape::to_string() const
    {
        std::string out = "(";
        for (size_t i = 0; i < dims_.size(); ++i)
        {
            if (i > 0)
            {
                out += ", ";
            }
            out += std::to_string(dims_[i]);
        }
        out += ")";
        return out;
    }

    std::string to_string(const Shape& shape)
    {
        return shape.to_string();
    }

    std::ostream& operator<<(std::ostream& os, const Shape& shape)
    {
        return os << shape.to_string();
    }
} // namespace veda::core
