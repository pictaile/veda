// E6.S1.T2 — nn::Embedding

#include "Embedding.h"
#include "Shape.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <stdexcept>
#include <string>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::nn::Embedding;

namespace
{
// rows: [0.1 0.2 0.3] [1.0 1.1 1.2] [2.0 2.1 2.2] [3.0 3.1 3.2]
Tensor table()
{
    Tensor t{Shape({4, 3})};
    const std::vector<float> rows = {0.1f, 0.2f, 0.3f, 1.0f, 1.1f, 1.2f,
                                     2.0f, 2.1f, 2.2f, 3.0f, 3.1f, 3.2f};
    for (size_t i = 0; i < rows.size(); ++i)
    {
        t.data()[i] = rows[i];
    }
    return t;
}
} // namespace

int main()
{
    // the worked example
    {
        const Embedding embedding(table());
        CHECK_EQ(embedding.vocab_size(), size_t{4});
        CHECK_EQ(embedding.hidden_size(), size_t{3});

        const Tensor out = embedding.forward({2, 0}, Shape({1, 2}));
        CHECK_EQ(out.shape(), Shape({1, 2, 3}));
        CHECK_EQ(out.at({0, 0, 0}), 2.0f);
        CHECK_EQ(out.at({0, 0, 2}), 2.2f);
        CHECK_EQ(out.at({0, 1, 0}), 0.1f);
        CHECK_EQ(out.at({0, 1, 2}), 0.3f);
        CHECK(out.is_contiguous());
    }

    // row extraction matches direct indexing of the weight tensor, for every id
    {
        const Tensor weights = table();
        const Embedding embedding(weights);

        for (size_t id = 0; id < 4; ++id)
        {
            const Tensor out = embedding.forward({static_cast<int64_t>(id)}, Shape({1, 1}));
            for (size_t d = 0; d < 3; ++d)
            {
                CHECK_EQ(out.at({0, 0, d}), weights.at({id, d}));
            }
        }
    }

    // AD2: the layer allocates no weights and copies none — it shares the storage
    {
        const Tensor weights = table();
        const Embedding embedding(weights);

        CHECK(embedding.table().storage() == weights.storage());
        CHECK(embedding.table().data() == weights.data());
        // two owners: the caller's tensor and the layer's — not two buffers
        CHECK_EQ(weights.storage().use_count(), long{2});

        // and the output is independent of the table: writing to it changes nothing
        Tensor out = embedding.forward({1}, Shape({1, 1}));
        out.data()[0] = 99.0f;
        CHECK_EQ(weights.at({1, 0}), 1.0f);
        CHECK(out.storage() != weights.storage());
    }

    // repeated ids give independent copies, not aliases of one row
    {
        const Embedding embedding(table());
        Tensor out = embedding.forward({1, 1, 1}, Shape({1, 3}));

        CHECK_EQ(out.at({0, 0, 0}), 1.0f);
        CHECK_EQ(out.at({0, 2, 0}), 1.0f);

        out.data()[0] = -5.0f;                  // write into the first copy
        CHECK_EQ(out.at({0, 1, 0}), 1.0f);      // the others are untouched
        CHECK_EQ(out.at({0, 2, 0}), 1.0f);
    }

    // the batch dimension is real, even though it is always 1 in practice (AD3)
    {
        const Embedding embedding(table());
        const Tensor out = embedding.forward({0, 1, 2, 3}, Shape({2, 2}));

        CHECK_EQ(out.shape(), Shape({2, 2, 3}));
        CHECK_EQ(out.at({0, 0, 0}), 0.1f);   // batch 0, position 0 -> row 0
        CHECK_EQ(out.at({0, 1, 0}), 1.0f);
        CHECK_EQ(out.at({1, 0, 0}), 2.0f);   // batch 1, position 0 -> row 2
        CHECK_EQ(out.at({1, 1, 2}), 3.2f);

        // a rank-1 id shape works too
        CHECK_EQ(embedding.forward({3}, Shape({1})).shape(), Shape({1, 3}));
    }

    // a table that is a view into something larger — which is what a real one will be
    {
        Tensor buffer{Shape({6, 3})};
        for (size_t i = 0; i < buffer.numel(); ++i)
        {
            buffer.data()[i] = static_cast<float>(i);
        }
        const Tensor slice = buffer.slice(0, 2, 3);   // rows 2..4, offset 6
        CHECK_EQ(slice.offset(), size_t{6});

        const Embedding embedding(slice);
        CHECK_EQ(embedding.vocab_size(), size_t{3});
        const Tensor out = embedding.forward({0, 2}, Shape({1, 2}));
        CHECK_EQ(out.at({0, 0, 0}), 6.0f);    // the slice's row 0 is the buffer's row 2
        CHECK_EQ(out.at({0, 1, 2}), 14.0f);
    }

    // a non-contiguous table: read through the strides, not the buffer
    {
        Tensor source{Shape({3, 4})};          // will be used transposed, as a [4, 3] table
        for (size_t i = 0; i < source.numel(); ++i)
        {
            source.data()[i] = static_cast<float>(i);
        }
        const Tensor transposed = source.transpose(0, 1);
        CHECK(!transposed.is_contiguous());

        const Embedding embedding(transposed);
        CHECK_EQ(embedding.vocab_size(), size_t{4});
        CHECK_EQ(embedding.hidden_size(), size_t{3});

        const Tensor out = embedding.forward({1}, Shape({1, 1}));
        for (size_t d = 0; d < 3; ++d)
        {
            CHECK_EQ(out.at({0, 0, d}), transposed.at({1, d}));
        }
        CHECK_EQ(out.at({0, 0, 0}), 1.0f);
        CHECK_EQ(out.at({0, 0, 1}), 5.0f);
    }

    // an empty sequence is legal
    {
        const Embedding embedding(table());
        const Tensor out = embedding.forward({}, Shape({1, 0}));
        CHECK_EQ(out.shape(), Shape({1, 0, 3}));
        CHECK_EQ(out.numel(), size_t{0});
    }

    // refusals
    {
        const Embedding embedding(table());

        // an id outside the vocabulary: the tokenizer and the model disagree (R2)
        CHECK_THROWS_AS(embedding.forward({4}, Shape({1, 1})), std::out_of_range);
        CHECK_THROWS_AS(embedding.forward({-1}, Shape({1, 1})), std::out_of_range);
        CHECK_THROWS_AS(embedding.forward({0, 99}, Shape({1, 2})), std::out_of_range);
        try
        {
            (void)embedding.forward({7}, Shape({1, 1}));
        }
        catch (const std::out_of_range& error)
        {
            const std::string message = error.what();
            CHECK(message.find("token id 7") != std::string::npos);
            CHECK(message.find("vocabulary of 4") != std::string::npos);
        }

        // an id count that disagrees with the shape
        CHECK_THROWS_AS(embedding.forward({1, 2}, Shape({1, 3})), std::invalid_argument);
        CHECK_THROWS_AS(embedding.forward({1, 2, 3}, Shape({1, 2})), std::invalid_argument);

        // the table must be rank 2
        CHECK_THROWS_AS(Embedding(Tensor{Shape({4})}), std::invalid_argument);
        CHECK_THROWS_AS(Embedding(Tensor{Shape({2, 2, 2})}), std::invalid_argument);
        CHECK_THROWS_AS(Embedding(Tensor{Shape({})}), std::invalid_argument);
    }

    // degenerate but legal tables
    {
        Tensor single{Shape({1, 1})};
        single.data()[0] = 7.0f;
        const Embedding one_token(single);
        CHECK_EQ(one_token.forward({0, 0}, Shape({1, 2})).at({0, 1, 0}), 7.0f);
        CHECK_THROWS_AS(one_token.forward({1}, Shape({1, 1})), std::out_of_range);

        // ids in descending order
        const Embedding embedding(table());
        const Tensor out = embedding.forward({3, 2, 1, 0}, Shape({1, 4}));
        CHECK_EQ(out.at({0, 0, 0}), 3.0f);
        CHECK_EQ(out.at({0, 3, 0}), 0.1f);
    }

    return VEDA_TEST_SUMMARY("EmbeddingTest");
}
