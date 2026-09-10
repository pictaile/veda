// E11.S2.T3 (temperature and truncation) and T4 (the three samplers)

#include "Sampling.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::generate::GreedySampler;
using veda::generate::TemperatureSampler;
using veda::generate::TopKSampler;
using veda::generate::TopPSampler;

namespace
{
Tensor filled(const std::vector<float>& values)
{
    Tensor t{Shape({values.size()})};
    for (size_t i = 0; i < values.size(); ++i)
    {
        t.data()[i] = values[i];
    }
    return t;
}

// How often each id comes up over many draws.
template <typename S>
std::map<int32_t, size_t> frequencies(S& sampler, const Tensor& logits, size_t draws)
{
    std::map<int32_t, size_t> counts;
    for (size_t i = 0; i < draws; ++i)
    {
        ++counts[sampler.sample(logits)];
    }
    return counts;
}
} // namespace

int main()
{
    // logits [3, 2, 1, 0] -> p = [0.6439, 0.2369, 0.0871, 0.0321]
    const Tensor logits = filled({3.0f, 2.0f, 1.0f, 0.0f});

    // --- the four limiting identities, which catch most truncation bugs ---------------------------
    {
        GreedySampler greedy;
        std::mt19937 engine(20260909);
        std::normal_distribution<float> values(0.0f, 3.0f);

        for (int trial = 0; trial < 30; ++trial)
        {
            std::vector<float> raw(1 + engine() % 32);
            for (float& value : raw)
            {
                value = values(engine);
            }
            const Tensor random_logits = filled(raw);
            const int32_t expected = greedy.sample(random_logits);

            TemperatureSampler zero_temperature(0.0f, 7);
            TopKSampler one(1, 1.0f, 7);
            TopPSampler tiny(0.0f, 1.0f, 7);

            CHECK_EQ(zero_temperature.sample(random_logits), expected);   // T = 0 is greedy
            CHECK_EQ(one.sample(random_logits), expected);                // k = 1 is greedy
            CHECK_EQ(tiny.sample(random_logits), expected);               // the nucleus is never empty
        }

        // k >= V truncates nothing: the same draws as plain temperature, id for id
        TopKSampler wide(1000, 1.0f, 42);
        TemperatureSampler plain(1.0f, 42);
        for (int i = 0; i < 50; ++i)
        {
            CHECK_EQ(wide.sample(logits), plain.sample(logits));
        }
    }

    // *** determinism: the same seed reproduces the same SEQUENCE, not just one draw ***
    {
        TemperatureSampler a(0.8f, 12345);
        TemperatureSampler b(0.8f, 12345);
        TemperatureSampler other(0.8f, 999);

        std::vector<int32_t> first;
        std::vector<int32_t> second;
        std::vector<int32_t> different;
        for (int i = 0; i < 40; ++i)
        {
            first.push_back(a.sample(logits));
            second.push_back(b.sample(logits));
            different.push_back(other.sample(logits));
        }
        CHECK(first == second);
        CHECK(first != different);
    }

    // *** temperature changes the gaps, never the order ***
    {
        TemperatureSampler sharp(0.3f, 1);
        TemperatureSampler neutral(1.0f, 1);
        TemperatureSampler flat(5.0f, 1);

        const size_t draws = 20000;
        const auto cold = frequencies(sharp, logits, draws);
        const auto warm = frequencies(neutral, logits, draws);
        const auto hot = frequencies(flat, logits, draws);

        // the top token dominates more as T falls, and less as it rises
        const double cold_share = static_cast<double>(cold.at(0)) / draws;
        const double warm_share = static_cast<double>(warm.at(0)) / draws;
        const double hot_share = static_cast<double>(hot.at(0)) / draws;

        CHECK(cold_share > warm_share);
        CHECK(warm_share > hot_share);
        CHECK(cold_share > 0.9);       // nearly greedy
        CHECK(hot_share < 0.45);       // approaching uniform (0.25)

        // and the ranking is unchanged at every temperature
        for (const auto& counts : {cold, warm, hot})
        {
            CHECK(counts.at(0) > counts.at(1));
            CHECK(counts.at(1) > counts.at(2));
        }
    }

    // *** the frequencies match the intended distribution ***
    // p = [0.6439, 0.2369, 0.0871, 0.0321] at T = 1
    {
        TemperatureSampler sampler(1.0f, 2024);
        const size_t draws = 40000;
        const auto counts = frequencies(sampler, logits, draws);

        CHECK_NEAR(static_cast<double>(counts.at(0)) / draws, 0.6439, 0.02);
        CHECK_NEAR(static_cast<double>(counts.at(1)) / draws, 0.2369, 0.02);
        CHECK_NEAR(static_cast<double>(counts.at(2)) / draws, 0.0871, 0.02);
        CHECK_NEAR(static_cast<double>(counts.at(3)) / draws, 0.0321, 0.02);
    }

    // *** top-k: only the k largest ever appear, and renormalisation is right ***
    {
        TopKSampler sampler(2, 1.0f, 77);
        const size_t draws = 20000;
        const auto counts = frequencies(sampler, logits, draws);

        CHECK_EQ(counts.size(), size_t{2});          // ids 2 and 3 never appear
        CHECK(counts.contains(0) && counts.contains(1));
        CHECK(!counts.contains(2));

        // renormalised: [0.6439, 0.2369] / 0.8808 = [0.731, 0.269]
        CHECK_NEAR(static_cast<double>(counts.at(0)) / draws, 0.731, 0.02);
        CHECK_NEAR(static_cast<double>(counts.at(1)) / draws, 0.269, 0.02);

        // k = 3 admits one more
        TopKSampler three(3, 1.0f, 77);
        CHECK_EQ(frequencies(three, logits, draws).size(), size_t{3});
    }

    // *** top-p adapts where top-k cannot ***
    {
        const size_t draws = 20000;

        // a confident model: p(top) = 0.97
        const Tensor peaked = filled({10.0f, 6.5f, 6.0f, 5.5f});
        TopPSampler nucleus_peaked(0.9f, 1.0f, 5);
        const auto peaked_counts = frequencies(nucleus_peaked, peaked, draws);
        CHECK_EQ(peaked_counts.size(), size_t{1});     // one token kept — the model is sure

        // an uncertain one: eight near-equal options
        const Tensor flat = filled({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
        TopPSampler nucleus_flat(0.9f, 1.0f, 5);
        const auto flat_counts = frequencies(nucleus_flat, flat, draws);
        CHECK(flat_counts.size() >= size_t{7});        // nearly all of them — the model is not

        // the same top-k cannot do both: k = 1 on the flat case keeps one
        TopKSampler fixed(1, 1.0f, 5);
        CHECK_EQ(frequencies(fixed, flat, draws).size(), size_t{1});
    }

    // the crossing token is included: cumulative 0.644, 0.881, 0.968 -> three ids at p = 0.9
    {
        TopPSampler sampler(0.9f, 1.0f, 11);
        const auto counts = frequencies(sampler, logits, 20000);
        CHECK_EQ(counts.size(), size_t{3});
        CHECK(!counts.contains(3));

        // renormalised over the nucleus: [0.665, 0.245, 0.090]
        CHECK_NEAR(static_cast<double>(counts.at(0)) / 20000, 0.665, 0.02);
        CHECK_NEAR(static_cast<double>(counts.at(2)) / 20000, 0.090, 0.02);

        // p = 1 keeps everything
        TopPSampler all(1.0f, 1.0f, 11);
        CHECK_EQ(frequencies(all, logits, 20000).size(), size_t{4});
    }

    // temperature is applied before truncation: a hotter sampler admits a wider nucleus
    {
        TopPSampler cold(0.9f, 0.5f, 3);
        TopPSampler hot(0.9f, 2.0f, 3);
        CHECK(frequencies(cold, logits, 20000).size() < frequencies(hot, logits, 20000).size());
    }

    // refusals
    {
        CHECK_THROWS_AS(TemperatureSampler(-1.0f, 0), std::invalid_argument);
        CHECK_THROWS_AS(TopKSampler(0, 1.0f, 0), std::invalid_argument);
        CHECK_THROWS_AS(TopPSampler(-0.1f, 1.0f, 0), std::invalid_argument);
        CHECK_THROWS_AS(TopPSampler(1.5f, 1.0f, 0), std::invalid_argument);
        CHECK_THROWS_AS(TopKSampler(5, -1.0f, 0), std::invalid_argument);

        const float nan = std::numeric_limits<float>::quiet_NaN();
        TemperatureSampler sampler(1.0f, 0);
        CHECK_THROWS_AS(sampler.sample(filled({1.0f, nan})), std::runtime_error);

        TopPSampler nucleus(0.9f, 1.0f, 0);
        CHECK_THROWS_AS(nucleus.sample(Tensor{Shape({0})}), std::invalid_argument);
        CHECK_THROWS_AS(nucleus.sample(Tensor{Shape({2, 2})}), std::invalid_argument);
    }

    // both accepted shapes, and a single-token vocabulary
    {
        TemperatureSampler sampler(1.0f, 0);
        Tensor row{Shape({1, 4})};
        for (size_t v = 0; v < 4; ++v)
        {
            row.data()[v] = logits.at({v});
        }
        CHECK(sampler.sample(row) >= 0);

        TopPSampler single(0.5f, 1.0f, 0);
        CHECK_EQ(single.sample(filled({7.0f})), 0);
    }

    return VEDA_TEST_SUMMARY("SamplingTest");
}
