#ifndef VEDA_MATMUL_H
#define VEDA_MATMUL_H

#include "Tensor.h"

namespace veda::ops
{

// How many threads a matmul may split its output rows across. 1 — the default — disables threading
// entirely, so a normal build behaves exactly as it did before E13.
//
// Rows are independent: row i reads all of A's row i and all of B, and writes only C's row i. So
// splitting them needs no synchronisation beyond the join, and — the point — leaves each output's
// summation order untouched, which keeps every result *bitwise* identical. Floating-point addition
// is not associative (E3.S1.T1), so an optimisation that reordered the sums would change the last
// bits and cost the exact tests written across twelve epics.
//
// The gain is bounded by memory bandwidth: inference reads every weight once per token, and four
// threads do not give four times the throughput on a kernel that is waiting on memory.
void set_thread_count(size_t threads);
size_t thread_count();

// Matrix multiplication.
//
//     C[i, j] = sum over k of  A[i, k] * B[k, j]
//     [..., M, K] x [..., K, N] -> [..., M, N]
//
// The last two dimensions are multiplied; every dimension to their left is a batch axis and is
// iterated. Batch axes broadcast by the usual rule, so [B,T,D] x [F,D] works with a weight that
// has no batch dimensions at all. Rank 2 is simply the case with an empty batch part.
//
// The only operation in Veda that is not linear in its element count: M*N*K multiply-adds for M*N
// outputs. Nearly all of a forward pass is spent here.
//
// Both operands may be non-contiguous — a transposed view arrives here constantly — so the loop
// reads them through their strides. Throws std::invalid_argument, naming both shapes, on a rank
// other than 2 or a mismatched inner dimension.
core::Tensor matmul(const core::Tensor& a, const core::Tensor& b);

// The same product with B already transposed:
//
//     C[i, j] = sum over k of  A[i, k] * B[j, k]
//     [..., M, K] x [..., N, K] -> [..., M, N]
//
// HuggingFace stores a linear weight as [out_features, in_features], so a Linear layer computes
// y = x * Wt. AD4 keeps that layout rather than transposing 2.4 GB at load, and this is the
// function that makes it work. The explicit _nt at every call site is a standing reminder of a
// convention that silently produces garbage when got wrong.
//
// Equivalent to matmul(a, b.transpose(0, 1)), but reads B along memory instead of across it.
core::Tensor matmul_nt(const core::Tensor& a, const core::Tensor& b);

} // namespace veda::ops

#endif //VEDA_MATMUL_H
