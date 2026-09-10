#include "Embedding.h"

#include "Profile.h"
#include "Storage.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace veda::nn
{

using core::Shape;
using core::Tensor;

Embedding::Embedding(Tensor table) : table_(std::move(table))
{
    if (table_.rank() != 2)
    {
        throw std::invalid_argument("nn::Embedding: the table must be rank 2 [V, D], got " +
                                    table_.shape().to_string());
    }
}

Tensor Embedding::forward(const std::vector<int64_t>& ids, const Shape& id_shape) const
{
    const profile::Scope scope("embedding");
    if (ids.size() != id_shape.size())
    {
        throw std::invalid_argument("nn::Embedding: " + std::to_string(ids.size()) +
                                    " ids for a shape " + id_shape.to_string() + " needing " +
                                    std::to_string(id_shape.size()));
    }

    const size_t vocabulary = vocab_size();
    const size_t hidden = hidden_size();

    std::vector<size_t> dims = id_shape.dims();
    dims.push_back(hidden);
    Tensor out{Shape(std::move(dims))};

    // Freshly allocated rather than a gather-view with stride 0: ids repeat, and two positions
    // holding the same id would alias one row. The next op to write there would corrupt both.
    float* destination = out.data();
    const float* source = table_.data();
    const size_t row_stride = table_.strides()[0];
    const size_t column_stride = table_.strides()[1];

    for (size_t i = 0; i < ids.size(); ++i)
    {
        const int64_t id = ids[i];
        // An id outside the vocabulary is not a wild pointer to catch later: it means the tokenizer
        // and the model disagree about the vocabulary, which is a specific and real bug (R2).
        if (id < 0 || static_cast<size_t>(id) >= vocabulary)
        {
            throw std::out_of_range("nn::Embedding: token id " + std::to_string(id) +
                                    " is outside a vocabulary of " + std::to_string(vocabulary));
        }

        // Read through the table's strides: it is a view into the weight buffer, and nothing in nn
        // may assume a layout that io did not promise.
        const size_t row_base = static_cast<size_t>(id) * row_stride;
        for (size_t d = 0; d < hidden; ++d)
        {
            destination[i * hidden + d] = source[row_base + d * column_stride];
        }
    }

    return out;
}

} // namespace veda::nn
