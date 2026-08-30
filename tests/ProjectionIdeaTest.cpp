// E6.S2.T3 — what a projection does, and where the parameters are (a Learn task)

#include "Matmul.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::ops::matmul;
using veda::ops::matmul_nt;

namespace
{
Tensor filled(const Shape& shape, const std::vector<float>& values)
{
    Tensor t{shape};
    for (size_t i = 0; i < values.size(); ++i)
    {
        t.data()[i] = values[i];
    }
    return t;
}
} // namespace

int main()
{
    // one row of the weight is one learned question asked of the input
    //   W = [[1,0,2],[0,1,1]]   x = [1,2,3]
    //   y0 = 1*1 + 0*2 + 2*3 = 7
    //   y1 = 0*1 + 1*2 + 1*3 = 5
    {
        const Tensor w = filled(Shape({2, 3}), {1, 0, 2, 0, 1, 1});
        const Tensor x = filled(Shape({1, 3}), {1, 2, 3});

        const Tensor y = matmul_nt(x, w);
        CHECK_EQ(y.shape(), Shape({1, 2}));
        CHECK_EQ(y.at({0, 0}), 7.0f);
        CHECK_EQ(y.at({0, 1}), 5.0f);

        // the shared dimension is the LAST of both operands — which is exactly matmul_nt
        CHECK_EQ(x.shape()[1], w.shape()[1]);
        CHECK_EQ(y.at({0, 0}), matmul(x, w.transpose(0, 1)).at({0, 0}));
    }

    // a projection is linear: no curvature, the origin stays put, and scaling passes through
    {
        const Tensor w = filled(Shape({2, 3}), {1, 0, 2, 0, 1, 1});
        const Tensor zero{Shape({1, 3})};
        CHECK_EQ(matmul_nt(zero, w).at({0, 0}), 0.0f);   // f(0) = 0

        const Tensor x = filled(Shape({1, 3}), {1, 2, 3});
        const Tensor doubled = filled(Shape({1, 3}), {2, 4, 6});
        CHECK_EQ(matmul_nt(doubled, w).at({0, 0}), 2.0f * matmul_nt(x, w).at({0, 0}));

        // and additive: f(a + b) = f(a) + f(b)
        const Tensor a = filled(Shape({1, 3}), {1, 0, 0});
        const Tensor b = filled(Shape({1, 3}), {0, 2, 3});
        CHECK_EQ(matmul_nt(a, w).at({0, 0}) + matmul_nt(b, w).at({0, 0}),
                 matmul_nt(x, w).at({0, 0}));
    }

    // where the parameters actually are, for Qwen3-0.6B
    {
        const size_t D = 1024;      // hidden
        const size_t F = 3072;      // ffn inner
        const size_t H = 16;        // query heads
        const size_t HKV = 8;       // kv heads
        const size_t DH = 128;      // head dim, as the config states it
        const size_t V = 151936;
        const size_t layers = 28;

        const size_t q = (H * DH) * D;
        const size_t k = (HKV * DH) * D;
        const size_t v = k;
        const size_t o = D * (H * DH);
        const size_t attention = q + k + v + o;

        const size_t gate = F * D;
        const size_t up = gate;
        const size_t down = D * F;
        const size_t ffn = gate + up + down;

        CHECK_EQ(q, size_t{2'097'152});
        CHECK_EQ(k, size_t{1'048'576});     // GQA: half of q, and so is v
        CHECK_EQ(k, q / 2);
        CHECK_EQ(attention + ffn, size_t{15'728'640});

        // the FFN holds most of the parameters while attention holds the concept
        CHECK(ffn > attention);
        CHECK_NEAR(static_cast<double>(ffn) / static_cast<double>(attention + ffn), 0.6, 0.01);

        const size_t all_layers = (attention + ffn) * layers;
        const size_t embeddings = V * D;    // also the LM head, tied
        const size_t total = all_layers + embeddings;

        // which is where "0.6B" comes from
        CHECK(total > 590'000'000 && total < 600'000'000);

        // the norms are invisible: 57 vectors of D floats
        const size_t norms = (2 * layers + 1) * D;
        CHECK_EQ(norms, size_t{58'368});
        CHECK(static_cast<double>(norms) / static_cast<double>(total) < 0.0001);
    }

    // batch dimensions pass through: the same weight serves every rank
    {
        const Tensor w{Shape({4, 3})};
        CHECK_EQ(matmul_nt(Tensor{Shape({8, 3})}, w).shape(), Shape({8, 4}));
        CHECK_EQ(matmul_nt(Tensor{Shape({1, 8, 3})}, w).shape(), Shape({1, 8, 4}));
        CHECK_EQ(matmul_nt(Tensor{Shape({1, 16, 8, 3})}, w).shape(), Shape({1, 16, 8, 4}));
    }

    return VEDA_TEST_SUMMARY("ProjectionIdeaTest");
}
