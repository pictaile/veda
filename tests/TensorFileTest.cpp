// E3.S2.T3 — the golden-file format and its reader

#include "Compare.h"
#include "Shape.h"
#include "TensorFile.h"
#include "TestSupport.h"
#include "Tensor.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using veda::core::Shape;
using veda::core::Tensor;
using veda::testing::compare_close;
using veda::testing::read_tensor;
using veda::testing::write_tensor;

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

std::filesystem::path scratch(const std::string& name)
{
    return std::filesystem::temp_directory_path() / ("veda_tensor_file_" + name);
}

// A round trip is a copy, not arithmetic — so it is compared exactly, with no tolerance.
bool identical(const Tensor& a, const Tensor& b)
{
    return compare_close(a, b, 0.0f, 0.0f).equal;
}
} // namespace

int main()
{
    // a round trip preserves shape and every value exactly
    {
        const auto path = scratch("basic.bin");
        const Tensor original = filled(Shape({2, 3}), {1, 2, 3, 4, 5, 6});

        write_tensor(path.string(), original);
        const Tensor loaded = read_tensor(path.string());

        CHECK_EQ(loaded.shape(), Shape({2, 3}));
        CHECK(identical(loaded, original));
        CHECK(loaded.is_contiguous());
        CHECK_EQ(loaded.at({1, 2}), 6.0f);

        // the byte layout from the format description: 8 + 4 + 4 + 8*2 + 4*6 = 56
        CHECK_EQ(std::filesystem::file_size(path), std::uintmax_t{56});
        std::filesystem::remove(path);
    }

    // values that must survive bit for bit, including negatives and awkward fractions
    {
        const auto path = scratch("values.bin");
        const Tensor original =
            filled(Shape({5}), {0.0f, -0.0f, 3.14159265f, -1e30f, 1.1754944e-38f});
        write_tensor(path.string(), original);
        const Tensor loaded = read_tensor(path.string());

        for (size_t i = 0; i < 5; ++i)
        {
            CHECK_EQ(loaded.at({i}), original.at({i}));
        }
        CHECK(std::signbit(loaded.at({1})));   // negative zero survives
        std::filesystem::remove(path);
    }

    // a rank-0 tensor: 16 bytes of header, one float of body
    {
        const auto path = scratch("scalar.bin");
        write_tensor(path.string(), filled(Shape({}), {7.5f}));
        const Tensor loaded = read_tensor(path.string());

        CHECK_EQ(loaded.shape().rank(), size_t{0});
        CHECK_EQ(loaded.numel(), size_t{1});
        CHECK_EQ(loaded.at({}), 7.5f);
        CHECK_EQ(std::filesystem::file_size(path), std::uintmax_t{20});
        std::filesystem::remove(path);
    }

    // a tensor with a zero dimension: header only, no body
    {
        const auto path = scratch("empty.bin");
        write_tensor(path.string(), Tensor{Shape({2, 0, 3})});
        const Tensor loaded = read_tensor(path.string());

        CHECK_EQ(loaded.shape(), Shape({2, 0, 3}));
        CHECK_EQ(loaded.numel(), size_t{0});
        CHECK_EQ(std::filesystem::file_size(path), std::uintmax_t{40});   // 16 + 8*3
        std::filesystem::remove(path);
    }

    // a non-contiguous view is stored in index order — strides are not part of the format
    {
        const auto path = scratch("view.bin");
        const Tensor source = filled(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
        const Tensor transposed = source.transpose(0, 1);   // (3,2), reads 1 4 2 5 3 6
        CHECK(!transposed.is_contiguous());

        write_tensor(path.string(), transposed);
        const Tensor loaded = read_tensor(path.string());

        CHECK_EQ(loaded.shape(), Shape({3, 2}));
        CHECK(loaded.is_contiguous());               // what comes back is a value, not a view
        CHECK(identical(loaded, transposed));
        CHECK_EQ(loaded.data()[1], 4.0f);            // stored in index order, not buffer order
        CHECK_EQ(loaded.data()[2], 2.0f);

        // and a sliced view with a non-zero offset
        const auto sliced_path = scratch("slice.bin");
        const Tensor window = source.slice(1, 1, 2);   // [[2,3],[5,6]]
        write_tensor(sliced_path.string(), window);
        CHECK(identical(read_tensor(sliced_path.string()), window));

        std::filesystem::remove(path);
        std::filesystem::remove(sliced_path);
    }

    // a big-ish tensor, to exercise more than a handful of elements
    {
        const auto path = scratch("large.bin");
        Tensor original{Shape({4, 8, 16})};
        for (size_t i = 0; i < original.numel(); ++i)
        {
            original.data()[i] = static_cast<float>(i) * 0.25f - 100.0f;
        }
        write_tensor(path.string(), original);
        CHECK(identical(read_tensor(path.string()), original));
        CHECK_EQ(std::filesystem::file_size(path),
                 std::uintmax_t{8 + 4 + 4 + 8 * 3 + 4 * 512});
        std::filesystem::remove(path);
    }

    // corruption: every case names the path and the reason
    {
        // a missing file
        CHECK_THROWS_AS(read_tensor(scratch("absent.bin").string()), std::runtime_error);
        try
        {
            (void)read_tensor(scratch("absent.bin").string());
        }
        catch (const std::runtime_error& error)
        {
            const std::string message = error.what();
            CHECK(message.find("absent.bin") != std::string::npos);
            CHECK(message.find("cannot be opened") != std::string::npos);
        }

        // wrong magic
        {
            const auto path = scratch("notveda.bin");
            std::ofstream out(path, std::ios::binary);
            const std::string content = "NOTATENSORFILE__________";
            out.write(content.data(), static_cast<std::streamsize>(content.size()));
            out.close();
            CHECK_THROWS_AS(read_tensor(path.string()), std::runtime_error);
            try
            {
                (void)read_tensor(path.string());
            }
            catch (const std::runtime_error& error)
            {
                CHECK(std::string(error.what()).find("bad magic") != std::string::npos);
            }
            std::filesystem::remove(path);
        }

        // a file shorter than the magic
        {
            const auto path = scratch("tiny.bin");
            std::ofstream out(path, std::ios::binary);
            out.write("VEDA", 4);
            out.close();
            CHECK_THROWS_AS(read_tensor(path.string()), std::runtime_error);
            std::filesystem::remove(path);
        }

        // a truncated body: a good file with its last bytes chopped off
        {
            const auto path = scratch("truncated.bin");
            write_tensor(path.string(), filled(Shape({2, 3}), {1, 2, 3, 4, 5, 6}));
            std::filesystem::resize_file(path, 50);   // 56 - 6
            CHECK_THROWS_AS(read_tensor(path.string()), std::runtime_error);
            try
            {
                (void)read_tensor(path.string());
            }
            catch (const std::runtime_error& error)
            {
                const std::string message = error.what();
                CHECK(message.find("body is") != std::string::npos);
                CHECK(message.find("(2, 3)") != std::string::npos);
            }
            std::filesystem::remove(path);
        }

        // a body longer than the header implies is corrupt too
        {
            const auto path = scratch("overlong.bin");
            write_tensor(path.string(), filled(Shape({2}), {1, 2}));
            std::ofstream out(path, std::ios::binary | std::ios::app);
            const float extra = 3.0f;
            out.write(reinterpret_cast<const char*>(&extra), sizeof(extra));
            out.close();
            CHECK_THROWS_AS(read_tensor(path.string()), std::runtime_error);
            std::filesystem::remove(path);
        }

        // an unknown dtype code: byte 8 of a good file set to 1
        {
            const auto path = scratch("dtype.bin");
            write_tensor(path.string(), filled(Shape({2}), {1, 2}));
            std::fstream patch(path, std::ios::binary | std::ios::in | std::ios::out);
            patch.seekp(8);
            const char one = 1;
            patch.write(&one, 1);
            patch.close();
            CHECK_THROWS_AS(read_tensor(path.string()), std::runtime_error);
            try
            {
                (void)read_tensor(path.string());
            }
            catch (const std::runtime_error& error)
            {
                CHECK(std::string(error.what()).find("unsupported dtype") != std::string::npos);
            }
            std::filesystem::remove(path);
        }
    }

    // the intended use, end to end: dump a reference value, compare a computed one against it
    {
        const auto path = scratch("reference.bin");
        write_tensor(path.string(), filled(Shape({2, 2}), {19, 22, 43, 50}));

        const Tensor expected = read_tensor(path.string());
        const Tensor actual = filled(Shape({2, 2}), {19.000002f, 22.0f, 43.0f, 50.000004f});
        CHECK(compare_close(actual, expected, 1e-5f, 1e-6f).equal);
        CHECK(!compare_close(actual, expected, 0.0f, 0.0f).equal);
        std::filesystem::remove(path);
    }

    return VEDA_TEST_SUMMARY("TensorFileTest");
}
