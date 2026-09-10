#include "Matmul.h"

#include "Broadcast.h"
#include "Shape.h"
#include "Storage.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace veda::ops
{

using core::Shape;
using core::Tensor;

namespace
{
size_t threads_ = 1;

// Below this many multiply-adds a matmul stays single-threaded: spawning threads for a 2x2 costs
// more than it saves. Chosen by measurement, not intuition.
constexpr size_t threading_threshold = 1 << 16;
} // namespace

void set_thread_count(size_t threads)
{
    threads_ = threads == 0 ? 1 : threads;
}

size_t thread_count()
{
    return threads_;
}

namespace
{

// One matrix product, written into a caller-provided contiguous destination. Both operands are
// described by two strides each, so matmul and matmul_nt — and every batch element of either —
// share this loop and cannot drift apart.
// A rectangle of the output: rows [first_row, last_row) and columns [first_column, last_column).
// Every output still sums k in ascending order, so the results are bitwise identical however the
// rectangle is carved up.
void multiply_block(const float* left, size_t a_row, size_t a_column,
                    const float* right, size_t b_shared, size_t b_column,
                    float* destination, size_t first_row, size_t last_row, size_t first_column,
                    size_t last_column, size_t k, size_t n)
{
    for (size_t i = first_row; i < last_row; ++i)
    {
        const size_t row_base = i * a_row;   // hoisted: independent of j and of the inner loop

        for (size_t j = first_column; j < last_column; ++j)
        {
            const size_t column_base = j * b_column;

            // The accumulator is a local, not destination[...]: accumulating into memory is both
            // slower and harder to read. It stays fp32 throughout (AD1) — and the summation order
            // is why E3 compares to a tolerance rather than exactly.
            //
            // With k == 0 the empty sum is 0, which is the right answer, not an error.
            float sum = 0.0f;
            for (size_t step = 0; step < k; ++step)
            {
                sum += left[row_base + step * a_column] * right[column_base + step * b_shared];
            }

            destination[i * n + j] = sum;
        }
    }
}

void multiply_into(const float* left, size_t a_row, size_t a_column,
                   const float* right, size_t b_shared, size_t b_column,
                   float* destination, size_t m, size_t k, size_t n)
{
    // Split along whichever output axis is long enough to divide.
    //
    // The profile made this necessary: during generation the activation is a single token, so the
    // output has ONE row, and splitting rows parallelises nothing. Only the prefill has rows to
    // divide. Splitting columns works in both cases and keeps each output's summation order.
    const bool split_rows = m >= threads_;
    const size_t divisible = split_rows ? m : n;
    const size_t threads = std::min(threads_, divisible);

    if (threads <= 1 || m * n * k < threading_threshold)
    {
        multiply_block(left, a_row, a_column, right, b_shared, b_column, destination, 0, m, 0, n, k,
                       n);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(threads - 1);
    const size_t per_thread = (divisible + threads - 1) / threads;

    auto run_slice = [&](size_t first, size_t last) {
        if (split_rows)
        {
            multiply_block(left, a_row, a_column, right, b_shared, b_column, destination, first,
                           last, 0, n, k, n);
        }
        else
        {
            multiply_block(left, a_row, a_column, right, b_shared, b_column, destination, 0, m,
                           first, last, k, n);
        }
    };

    for (size_t t = 1; t < threads; ++t)
    {
        const size_t first = std::min(t * per_thread, divisible);
        const size_t last = std::min(first + per_thread, divisible);
        if (first >= last)
        {
            break;
        }
        workers.emplace_back([=, &run_slice] { run_slice(first, last); });
    }

    run_slice(0, std::min(per_thread, divisible));

    for (std::thread& worker : workers)
    {
        worker.join();
    }
}

// Everything to the left of the last two dimensions.
Shape batch_shape_of(const Shape& shape)
{
    const std::vector<size_t>& dims = shape.dims();
    return Shape(std::vector<size_t>(dims.begin(), dims.end() - 2));
}

// The tensor's batch strides, read as if its batch part had the target shape: zero wherever an
// axis is repeated, and zero for the leading axes it does not have at all. The same stride-0
// mechanism as broadcast_to, applied to the batch part only.
std::vector<size_t> batch_strides_for(const Tensor& tensor, const Shape& target)
{
    const size_t batch_rank = tensor.rank() - 2;
    std::vector<size_t> strides(target.rank(), 0);

    for (size_t i = 0; i < batch_rank; ++i)
    {
        const size_t from_end = batch_rank - 1 - i;
        strides[target.rank() - 1 - i] = tensor.shape()[from_end] == 1 ? 0 : tensor.strides()[from_end];
    }
    return strides;
}

void advance(std::vector<size_t>& index, const Shape& shape)
{
    for (size_t k = shape.rank(); k-- > 0;)
    {
        if (++index[k] < shape[k])
        {
            return;
        }
        index[k] = 0;
    }
}

// matmul and matmul_nt differ only in which of B's two trailing dimensions is the shared one.
// Rank 2 is the degenerate case of this, with an empty batch part — not a separate code path.
Tensor run(const char* op, const Tensor& a, const Tensor& b, bool b_transposed)
{
    if (a.rank() < 2 || b.rank() < 2)
    {
        throw std::invalid_argument(
            std::string(op) + ": both operands must be at least rank 2, got " +
            a.shape().to_string() + " and " + b.shape().to_string());
    }

    const size_t m = a.shape()[a.rank() - 2];
    const size_t k = a.shape()[a.rank() - 1];

    // B is [K, N] for matmul and [N, K] for matmul_nt: the two trailing roles swap, and with them
    // the two strides.
    const size_t shared_of_b = b_transposed ? b.shape()[b.rank() - 1] : b.shape()[b.rank() - 2];
    const size_t n = b_transposed ? b.shape()[b.rank() - 2] : b.shape()[b.rank() - 1];
    const size_t b_shared_stride = b_transposed ? b.strides()[b.rank() - 1] : b.strides()[b.rank() - 2];
    const size_t b_column_stride = b_transposed ? b.strides()[b.rank() - 2] : b.strides()[b.rank() - 1];

    if (k != shared_of_b)
    {
        throw std::invalid_argument(
            std::string(op) + ": shared dimensions do not match: " + a.shape().to_string() + " x " +
            b.shape().to_string() + " (" + std::to_string(k) + " against " +
            std::to_string(shared_of_b) + ")");
    }

    const Shape a_batch = batch_shape_of(a.shape());
    const Shape b_batch = batch_shape_of(b.shape());

    Shape batch({});
    try
    {
        batch = broadcast_shape(a_batch, b_batch);
    }
    catch (const std::invalid_argument&)
    {
        throw std::invalid_argument(
            std::string(op) + ": batch dimensions do not broadcast: " + a.shape().to_string() +
            " x " + b.shape().to_string());
    }

    std::vector<size_t> result_dims = batch.dims();
    result_dims.push_back(m);
    result_dims.push_back(n);
    Tensor out{Shape(std::move(result_dims))};

    if (out.numel() == 0)
    {
        return out;
    }

    const std::vector<size_t> a_batch_strides = batch_strides_for(a, batch);
    const std::vector<size_t> b_batch_strides = batch_strides_for(b, batch);

    const float* left = a.data();
    const float* right = b.data();
    const size_t a_row = a.strides()[a.rank() - 2];
    const size_t a_column = a.strides()[a.rank() - 1];

    // The output is contiguous, so batch element e occupies destination[e * m * n ...]. The batch
    // walk visits the elements in that same row-major order.
    std::vector<size_t> index(batch.rank(), 0);
    for (size_t e = 0; e < batch.size(); ++e)
    {
        size_t a_offset = 0;
        size_t b_offset = 0;
        for (size_t axis = 0; axis < batch.rank(); ++axis)
        {
            a_offset += index[axis] * a_batch_strides[axis];
            b_offset += index[axis] * b_batch_strides[axis];
        }

        multiply_into(left + a_offset, a_row, a_column,
                      right + b_offset, b_shared_stride, b_column_stride,
                      out.data() + e * m * n, m, k, n);

        advance(index, batch);
    }

    return out;
}

} // namespace

Tensor matmul(const Tensor& a, const Tensor& b)
{
    return run("matmul", a, b, false);
}

Tensor matmul_nt(const Tensor& a, const Tensor& b)
{
    return run("matmul_nt", a, b, true);
}

} // namespace veda::ops
