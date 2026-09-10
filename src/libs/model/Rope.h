#ifndef VEDA_ROPE_H
#define VEDA_ROPE_H

#include "Tensor.h"

#include <cstddef>

namespace veda::model
{

// Rotary position encoding.
//
// Attention without it is permutation-invariant: score[t, s] = q_t . k_s depends on the two vectors
// and nothing else, so shuffling the input shuffles the output and "the cat chased the dog" and
// "the dog chased the cat" are nearly the same sentence (E7's last test proves it).
//
// RoPE splits each head's vector into Dh/2 pairs, treats each as a point in a plane, and rotates
// pair i by an angle proportional to the position:
//
//     omega_i = 1 / theta^(2i / Dh)        angle at position p = p * omega_i
//
// The point is what that does to a score. Rotations compose and a dot product is invariant to
// rotating both vectors together, so
//
//     R(theta_q) q . R(theta_k) k = q . R(theta_k - theta_q) k
//
// and the score depends on the DIFFERENCE of the positions. The model never learns "position 5"; it
// learns "three tokens back", and absolute position cancels out of the arithmetic entirely.
//
// Applied to Q and K only, never to V: V carries what a position contributes, and rotating it would
// mix position into the payload. Q and K decide how much, and that is where position belongs.

// cos and sin of every angle, [positions, head_dim], with each angle repeated across both halves of
// the head dimension. They depend only on position and head dimension — not on the head, the layer,
// or the data — so they are built once per sequence length and reused everywhere.
//
// first_position shifts every angle: with a KV cache (E12) a new query sits at the end of the
// history rather than at 0, and the same function serves prefill and generation.
//
// Throws if head_dim is odd — there are no odd pairs.
struct RopeTables
{
    core::Tensor cosines;
    core::Tensor sines;
};

RopeTables rope_tables(size_t positions, size_t head_dim, float theta, size_t first_position = 0);

// [.., T, Dh] -> [.., T, Dh], rotated.
//
// *** The pairing convention. *** The paper rotates adjacent components (x0,x1), (x2,x3), ... The
// HuggingFace implementation Qwen3 was trained with rotates halves — it pairs x_i with x_{i+Dh/2}:
//
//     rotate_half(x) = concat(-x[Dh/2:], x[:Dh/2])
//     out            = x * cos + rotate_half(x) * sin
//
// The two are the same rotation over a permuted set of channels. Mathematically equivalent — norm
// preservation, position 0, relative distance all hold either way — but the weights were trained
// with one of them and the channels are laid out to match. Choosing the other gives vectors of the
// right length with the right distance property and the wrong value in every channel: the model
// runs, generates, and is subtly wrong.
//
// Veda implements the half-split convention, because the goal is to reproduce Qwen3's outputs
// rather than the paper's notation. Only the reference comparison (E3) can confirm it.
core::Tensor apply_rope(const core::Tensor& x, const RopeTables& tables);

} // namespace veda::model

#endif //VEDA_ROPE_H
