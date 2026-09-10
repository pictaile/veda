// E13.S1.T1 (measurement) and T2 (optimisation)

#include "Matmul.h"
#include "Profile.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <random>
#include <string>
#include <thread>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::profile::Profile;
using veda::profile::Scope;

namespace
{
Tensor random_tensor(const Shape& shape, std::mt19937& engine)
{
    std::normal_distribution<float> values(0.0f, 1.0f);
    Tensor t{shape};
    for (size_t i = 0; i < shape.size(); ++i)
    {
        t.data()[i] = values(engine);
    }
    return t;
}

bool identical(const Tensor& a, const Tensor& b)
{
    if (a.shape() != b.shape())
    {
        return false;
    }
    for (size_t i = 0; i < a.numel(); ++i)
    {
        if (a.data()[i] != b.data()[i])
        {
            return false;
        }
    }
    return true;
}
} // namespace

int main()
{
    // --- T1: the profile ---------------------------------------------------------------------
    {
        Profile profile;
        CHECK(profile.empty());

        profile.record("attention", 3000);
        profile.record("attention", 1000);
        profile.record("feed_forward", 6000);

        CHECK_EQ(profile.calls("attention"), size_t{2});
        CHECK_EQ(profile.nanoseconds("attention"), uint64_t{4000});
        CHECK_EQ(profile.total_nanoseconds(), uint64_t{10000});
        CHECK_EQ(profile.calls("absent"), size_t{0});

        const std::string rendered = profile.to_string();
        CHECK(rendered.find("attention") != std::string::npos);
        CHECK(rendered.find("feed_forward") != std::string::npos);
        // sorted by total: the feed-forward's 60% comes first
        CHECK(rendered.find("feed_forward") < rendered.find("attention"));
        CHECK(rendered.find("60.0%") != std::string::npos);
        CHECK(rendered.find("40.0%") != std::string::npos);

        profile.reset();
        CHECK(profile.empty());
        CHECK_EQ(profile.total_nanoseconds(), uint64_t{0});
    }

    // *** instrumentation is off by default and costs nothing ***
    {
        veda::profile::active_profile().reset();
        CHECK(!veda::profile::profiling_enabled());

        {
            const Scope scope("ignored");
        }
        CHECK(veda::profile::active_profile().empty());

        veda::profile::set_profiling_enabled(true);
        {
            const Scope outer("outer");
            {
                const Scope inner("inner");   // nested scopes both record
            }
        }
        veda::profile::set_profiling_enabled(false);

        CHECK_EQ(veda::profile::active_profile().calls("outer"), size_t{1});
        CHECK_EQ(veda::profile::active_profile().calls("inner"), size_t{1});
        CHECK(veda::profile::active_profile().nanoseconds("outer") > 0);

        // and it stops recording again once disabled
        {
            const Scope scope("outer");
        }
        CHECK_EQ(veda::profile::active_profile().calls("outer"), size_t{1});
        veda::profile::active_profile().reset();
    }

    // --- T2: threading changes the time and nothing else --------------------------------------
    {
        CHECK_EQ(veda::ops::thread_count(), size_t{1});   // off by default

        std::mt19937 engine(20260910);
        const std::vector<std::vector<size_t>> shapes = {
            {64, 96, 128},    // a prefill-shaped matmul: many rows
            {1, 512, 512},    // a generation-shaped one: ONE row, so rows cannot be split
            {7, 33, 129},     // awkward sizes that do not divide by the thread count
            {2, 2, 2},        // below the threshold: stays single-threaded
        };

        for (const std::vector<size_t>& dims : shapes)
        {
            const Tensor a = random_tensor(Shape({dims[0], dims[1]}), engine);
            const Tensor b = random_tensor(Shape({dims[2], dims[1]}), engine);

            veda::ops::set_thread_count(1);
            const Tensor single = veda::ops::matmul_nt(a, b);

            for (const size_t threads : {2u, 4u, 8u})
            {
                veda::ops::set_thread_count(threads);
                const Tensor many = veda::ops::matmul_nt(a, b);

                // *** bitwise, not within a tolerance: the claim is that nothing changed ***
                CHECK(identical(single, many));
            }
            veda::ops::set_thread_count(1);
        }

        CHECK_EQ(veda::ops::thread_count(), size_t{1});
    }

    // the same for plain matmul, and for batched shapes
    {
        std::mt19937 engine(31415);
        const Tensor a = random_tensor(Shape({2, 40, 60}), engine);
        const Tensor b = random_tensor(Shape({2, 60, 80}), engine);

        veda::ops::set_thread_count(1);
        const Tensor single = veda::ops::matmul(a, b);
        veda::ops::set_thread_count(4);
        const Tensor threaded = veda::ops::matmul(a, b);
        veda::ops::set_thread_count(1);

        CHECK(identical(single, threaded));
    }

    // set_thread_count(0) is treated as 1 rather than as "no threads at all"
    {
        veda::ops::set_thread_count(0);
        CHECK_EQ(veda::ops::thread_count(), size_t{1});
    }

    // the arithmetic behind the ceiling: inference is memory-bandwidth-bound
    {
        const size_t parameters = 596'000'000;
        const size_t bytes_fp32 = parameters * 4;
        const size_t bytes_int4 = parameters / 2;

        CHECK(bytes_fp32 > 2'000'000'000);            // 2.4 GB read per token
        CHECK_EQ(bytes_fp32 / bytes_int4, size_t{8});  // what quantisation would save in traffic

        // at 25 GB/s that is the ceiling, whatever the arithmetic does
        const double seconds_per_token = static_cast<double>(bytes_fp32) / 25e9;
        CHECK(seconds_per_token > 0.09 && seconds_per_token < 0.11);
        CHECK_NEAR(1.0 / seconds_per_token, 10.5, 1.0);   // about ten tokens per second
    }

    return VEDA_TEST_SUMMARY("PerformanceTest");
}
