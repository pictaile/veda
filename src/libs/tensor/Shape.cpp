#include "Shape.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace veda::core
{
    namespace
    {
        // The product of the dimensions, starting from 1 so that the empty product falls out for
        // free: the size of a rank-0 shape is 1, not 0.
        //
        // The multiplication is checked rather than trusted. Shapes are built from weight-file
        // headers, and a corrupt one can claim any dimensions at all; an unchecked product would
        // wrap around to a small number and quietly under-allocate.
        size_t element_count(const std::vector<size_t>& dims)
        {
            size_t count = 1;
            for (const size_t dim : dims)
            {
                if (dim != 0 && count > std::numeric_limits<size_t>::max() / dim)
                {
                    std::string rendered;
                    for (size_t i = 0; i < dims.size(); ++i)
                    {
                        rendered += (i > 0 ? ", " : "") + std::to_string(dims[i]);
                    }
                    throw std::overflow_error(
                        "Shape: element count of (" + rendered + ") overflows size_t");
                }
                count *= dim;
            }
            return count;
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
