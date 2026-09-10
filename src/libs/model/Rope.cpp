#include "Rope.h"

#include "Shape.h"
#include "Storage.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace veda::model
{

using core::Shape;
using core::Tensor;

RopeTables rope_tables(size_t positions, size_t head_dim, float theta, size_t first_position)
{
    if (head_dim == 0 || head_dim % 2 != 0)
    {
        throw std::invalid_argument("model::rope_tables: head_dim must be even, got " +
                                    std::to_string(head_dim));
    }
    if (!(theta > 0.0f))
    {
        throw std::invalid_argument("model::rope_tables: theta must be positive");
    }

    const size_t pairs = head_dim / 2;

    Tensor cosines{Shape({positions, head_dim})};
    Tensor sines{Shape({positions, head_dim})};

    float* cosine_data = cosines.data();
    float* sine_data = sines.data();

    for (size_t p = 0; p < positions; ++p)
    {
        const double position = static_cast<double>(first_position + p);

        for (size_t i = 0; i < pairs; ++i)
        {
            // Geometrically spaced from 1 (fast) down to 1/theta (slow): the fast pairs resolve
            // nearby distances precisely, the slow ones stay distinguishable over thousands of
            // positions. Computed in double — at theta = 1e6 the smallest frequency is 1e-6.
            const double frequency =
                std::pow(static_cast<double>(theta),
                         -2.0 * static_cast<double>(i) / static_cast<double>(head_dim));
            const double angle = position * frequency;

            const float cosine = static_cast<float>(std::cos(angle));
            const float sine = static_cast<float>(std::sin(angle));

            // Each angle appears in both halves, because the pairing is (i, i + Dh/2).
            cosine_data[p * head_dim + i] = cosine;
            cosine_data[p * head_dim + i + pairs] = cosine;
            sine_data[p * head_dim + i] = sine;
            sine_data[p * head_dim + i + pairs] = sine;
        }
    }

    return RopeTables{std::move(cosines), std::move(sines)};
}

Tensor apply_rope(const Tensor& x, const RopeTables& tables)
{
    if (x.rank() < 2)
    {
        throw std::invalid_argument("model::apply_rope: expected [.., T, Dh], got " +
                                    x.shape().to_string());
    }

    const size_t positions = x.shape()[x.rank() - 2];
    const size_t head_dim = x.shape()[x.rank() - 1];

    if (tables.cosines.shape() != Shape({positions, head_dim}))
    {
        throw std::invalid_argument("model::apply_rope: tables are " +
                                    tables.cosines.shape().to_string() + ", input needs (" +
                                    std::to_string(positions) + ", " + std::to_string(head_dim) +
                                    ")");
    }

    Tensor out{x.shape()};
    if (x.numel() == 0)
    {
        return out;
    }

    const size_t pairs = head_dim / 2;
    const size_t lanes = x.numel() / (positions * head_dim);   // every batch and head

    const float* source = x.storage()->data();
    const float* cosine_data = tables.cosines.data();
    const float* sine_data = tables.sines.data();
    float* destination = out.data();

    // The input arrives straight from split_heads and is strided, so it is read through its strides
    // rather than as a flat buffer.
    const std::vector<size_t>& strides = x.strides();
    const size_t position_stride = strides[x.rank() - 2];
    const size_t channel_stride = strides[x.rank() - 1];

    std::vector<size_t> leading(x.rank() >= 2 ? x.rank() - 2 : 0, 0);
    for (size_t lane = 0; lane < lanes; ++lane)
    {
        size_t base = x.offset();
        for (size_t k = 0; k < leading.size(); ++k)
        {
            base += leading[k] * strides[k];
        }
        const size_t out_base = lane * positions * head_dim;

        for (size_t p = 0; p < positions; ++p)
        {
            const size_t row = base + p * position_stride;
            const size_t out_row = out_base + p * head_dim;

            for (size_t i = 0; i < pairs; ++i)
            {
                const float first = source[row + i * channel_stride];
                const float second = source[row + (i + pairs) * channel_stride];

                const float cosine = cosine_data[p * head_dim + i];
                const float sine = sine_data[p * head_dim + i];

                // out = x * cos + rotate_half(x) * sin, where rotate_half puts -second in the first
                // half and first in the second — the plane rotation, written out.
                destination[out_row + i] = first * cosine - second * sine;
                destination[out_row + i + pairs] = second * cosine + first * sine;
            }
        }

        for (size_t k = leading.size(); k-- > 0;)
        {
            if (++leading[k] < x.shape()[k])
            {
                break;
            }
            leading[k] = 0;
        }
    }

    return out;
}

} // namespace veda::model
