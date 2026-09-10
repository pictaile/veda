// E16.S1.T1 — the corpus: text in, batches out, and why the text labels itself

#include "ByteLevel.h"
#include "Corpus.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tokenizer.h"

#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using veda::core::Shape;
using veda::data::Batch;
using veda::data::Corpus;
using veda::data::Split;
using veda::tokenizer::byte_tokenizer;

namespace
{
std::vector<int64_t> consecutive(size_t count, int64_t from = 0)
{
    std::vector<int64_t> ids(count);
    for (size_t i = 0; i < count; ++i)
    {
        ids[i] = from + static_cast<int64_t>(i);
    }
    return ids;
}
} // namespace

int main()
{
    // --- the byte tokenizer: 256 ids, and nothing is ever unknown -----------------------------------
    {
        const auto tokenizer = byte_tokenizer();

        const std::vector<std::string> samples = {
            "hello",
            "Привіт, світе",
            "  double  spaces\tand\ttabs\n",
            std::string("a\0b", 3),      // a NUL in the middle
            "\xff\xfe\x01",              // bytes that are not valid UTF-8 on their own
            "",
        };

        bool round_trips = true;
        for (const std::string& sample : samples)
        {
            const std::vector<int32_t> ids = tokenizer.encode(sample);
            round_trips = round_trips && ids.size() == sample.size();   // one id per byte, no merges
            round_trips = round_trips && tokenizer.decode(ids) == sample;
        }
        CHECK(round_trips);

        // the ids are the byte values themselves
        const std::vector<int32_t> abc = tokenizer.encode("abc");
        CHECK_EQ(abc[0], 97);
        CHECK_EQ(abc[1], 98);
        CHECK_EQ(abc[2], 99);

        // Cyrillic is two ids per character — the cost of bytes, stated rather than hidden
        CHECK_EQ(tokenizer.encode("сон").size(), size_t{6});
    }

    // *** the targets are the inputs shifted by one ***
    {
        // A corpus of consecutive integers makes an off-by-one visible at a glance: whatever the
        // window, targets[i] must be inputs[i] + 1.
        Corpus corpus(consecutive(200), 0.0, 12345);
        const Batch batch = corpus.next_batch(4, 8, Split::Train);

        CHECK_EQ(batch.shape, Shape({4, 8}));
        CHECK_EQ(batch.inputs.size(), size_t{32});
        CHECK_EQ(batch.targets.size(), size_t{32});

        bool shifted = true;
        for (size_t i = 0; i < batch.inputs.size(); ++i)
        {
            shifted = shifted && batch.targets[i] == batch.inputs[i] + 1;
        }
        CHECK(shifted);

        // and each row is a contiguous run, not a shuffle of positions
        bool contiguous = true;
        for (size_t row = 0; row < 4; ++row)
        {
            for (size_t t = 1; t < 8; ++t)
            {
                contiguous = contiguous &&
                             batch.inputs[row * 8 + t] == batch.inputs[row * 8 + t - 1] + 1;
            }
        }
        CHECK(contiguous);
    }

    // the worked example from the task document
    {
        Corpus corpus(consecutive(10, 97), 0.2, 7);   // "abcdefghij"
        CHECK_EQ(corpus.boundary(), size_t{8});
        CHECK_EQ(corpus.size(Split::Train), size_t{8});
        CHECK_EQ(corpus.size(Split::Validation), size_t{2});

        // a window of 3 needs 4 ids, so the last legal offset in the training split is 8 - 4 = 4
        bool inside = true;
        for (int attempt = 0; attempt < 50; ++attempt)
        {
            const Batch batch = corpus.next_batch(2, 3, Split::Train);
            for (size_t row = 0; row < 2; ++row)
            {
                inside = inside && batch.inputs[row * 3] >= 97 && batch.inputs[row * 3] <= 97 + 4;
            }
        }
        CHECK(inside);
    }

    // *** no validation window can overlap a training window ***
    {
        // The property this design exists for, and the one a leaky implementation still passes
        // every other test with.
        Corpus corpus(consecutive(1000), 0.25, 999);
        CHECK_EQ(corpus.boundary(), size_t{750});

        std::set<int64_t> seen_in_training;
        for (int attempt = 0; attempt < 200; ++attempt)
        {
            const Batch batch = corpus.next_batch(4, 16, Split::Train);
            for (const int64_t id : batch.inputs)
            {
                seen_in_training.insert(id);
            }
            for (const int64_t id : batch.targets)
            {
                seen_in_training.insert(id);
            }
        }

        bool inside_region = true;
        bool disjoint = true;
        for (int attempt = 0; attempt < 200; ++attempt)
        {
            const Batch batch = corpus.next_batch(4, 16, Split::Validation);
            for (const int64_t id : batch.inputs)
            {
                inside_region = inside_region && id >= 750;
                disjoint = disjoint && seen_in_training.find(id) == seen_in_training.end();
            }
        }
        CHECK(inside_region);
        CHECK(disjoint);

        // and training really did cover most of its own region, so the check above had teeth
        CHECK(seen_in_training.size() > 700);
    }

    // *** random offsets, and the same seed reproducing them exactly ***
    {
        Corpus first(consecutive(500), 0.0, 4242);
        Corpus second(consecutive(500), 0.0, 4242);
        Corpus different(consecutive(500), 0.0, 4243);

        const Batch a1 = first.next_batch(3, 10, Split::Train);
        const Batch a2 = first.next_batch(3, 10, Split::Train);
        const Batch b1 = second.next_batch(3, 10, Split::Train);
        const Batch c1 = different.next_batch(3, 10, Split::Train);

        CHECK(a1.inputs == b1.inputs);        // same seed, same batches, in the same order
        CHECK(a1.inputs != a2.inputs);        // consecutive batches differ — offsets are random
        CHECK(a1.inputs != c1.inputs);        // a different seed gives a different run

        // the second batch of the reproduced run matches too, not merely the first
        CHECK(a2.inputs == second.next_batch(3, 10, Split::Train).inputs);

        // rows within one batch are drawn independently
        const Batch spread = first.next_batch(8, 4, Split::Train);
        bool rows_differ = false;
        for (size_t row = 1; row < 8; ++row)
        {
            if (spread.inputs[row * 4] != spread.inputs[0])
            {
                rows_differ = true;
            }
        }
        CHECK(rows_differ);
    }

    // --- text in, through the tokenizer ---------------------------------------------------------------
    {
        const auto tokenizer = byte_tokenizer();
        const std::string text = "the quick brown fox jumps over the lazy dog, again and again";
        Corpus corpus = Corpus::from_text(text, tokenizer, 0.1, 2026);

        CHECK_EQ(corpus.ids().size(), text.size());
        CHECK_EQ(corpus.ids()[0], static_cast<int64_t>('t'));

        const Batch batch = corpus.next_batch(2, 8, Split::Train);

        // a row decodes back to a readable substring of the original — the round trip that matters
        std::vector<int32_t> row(batch.inputs.begin(), batch.inputs.begin() + 8);
        const std::string decoded = tokenizer.decode(row);
        CHECK_EQ(decoded.size(), size_t{8});
        CHECK(text.find(decoded) != std::string::npos);
    }

    // --- refusals and edges -----------------------------------------------------------------------------
    {
        Corpus small(consecutive(20), 0.5, 1);
        CHECK_EQ(small.size(Split::Train), size_t{10});

        // a window of 10 needs 11 ids, and the split holds 10
        CHECK_THROWS_AS(small.next_batch(1, 10, Split::Train), std::invalid_argument);
        // 9 needs 10, which fits exactly
        (void)small.next_batch(1, 9, Split::Train);

        CHECK_THROWS_AS(small.next_batch(0, 4, Split::Train), std::invalid_argument);
        CHECK_THROWS_AS(small.next_batch(2, 0, Split::Train), std::invalid_argument);
        CHECK_THROWS_AS(Corpus(consecutive(10), -0.1, 1), std::invalid_argument);
        CHECK_THROWS_AS(Corpus(consecutive(10), 1.5, 1), std::invalid_argument);

        try
        {
            (void)small.next_batch(1, 10, Split::Train);
        }
        catch (const std::invalid_argument& error)
        {
            const std::string message = error.what();
            CHECK(message.find("training") != std::string::npos);
            CHECK(message.find("holds 10") != std::string::npos);
            CHECK(message.find("needs 11") != std::string::npos);
        }

        // no validation at all is legal, and asking for it says so clearly
        Corpus everything(consecutive(50), 0.0, 1);
        CHECK_EQ(everything.size(Split::Validation), size_t{0});
        CHECK_THROWS_AS(everything.next_batch(1, 4, Split::Validation), std::invalid_argument);

        // and the other extreme
        Corpus nothing(consecutive(50), 1.0, 1);
        CHECK_EQ(nothing.size(Split::Train), size_t{0});
        CHECK_EQ(nothing.size(Split::Validation), size_t{50});
    }

    return VEDA_TEST_SUMMARY("CorpusTest");
}
