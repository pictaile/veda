#include "Elementwise.h"

#include "Broadcast.h"
#include "Shape.h"
#include "Storage.h"
#include "Walk.h"

#include <vector>

namespace veda::ops
{

using core::Shape;
using core::Tensor;

namespace
{

using detail::advance;
using detail::read;

// The shared walk. Both binary ops differ only in the operation applied, so the loop exists once:
// two hand-written copies would diverge the first time one of them was fixed.
template <typename Op>
Tensor zip(const Tensor& a, const Tensor& b, Op op)
{
    const Shape shape = broadcast_shape(a.shape(), b.shape());
    const Tensor lhs = broadcast_to(a, shape);
    const Tensor rhs = broadcast_to(b, shape);

    // Freshly allocated: the inputs may be views over the same storage — or over each other — so
    // writing into either would corrupt the computation halfway through.
    Tensor out{shape};
    if (shape.size() == 0)
    {
        return out;
    }

    // The inputs are walked through their strides (they are rarely contiguous after E1); the
    // output was just allocated, so it is contiguous and can be filled straight through.
    std::vector<size_t> index(shape.rank(), 0);
    float* destination = out.data();
    for (size_t i = 0; i < shape.size(); ++i)
    {
        destination[i] = op(read(lhs, index), read(rhs, index));
        advance(index, shape);
    }
    return out;
}

template <typename Op>
Tensor map(const Tensor& a, float scalar, Op op)
{
    return detail::map_elements(a, [&](float x) { return op(x, scalar); });
}

} // namespace

Tensor add(const Tensor& a, const Tensor& b)
{
    return zip(a, b, [](float x, float y) { return x + y; });
}

Tensor mul(const Tensor& a, const Tensor& b)
{
    return zip(a, b, [](float x, float y) { return x * y; });
}

Tensor add(const Tensor& a, float scalar)
{
    return map(a, scalar, [](float x, float s) { return x + s; });
}

Tensor mul(const Tensor& a, float scalar)
{
    return map(a, scalar, [](float x, float s) { return x * s; });
}

} // namespace veda::ops
