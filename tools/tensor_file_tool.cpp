// A tiny bridge for the cross-language round-trip test (E3.S2.T4).
//
//     tensor_file_tool write  <path>     writes the shared fixture
//     tensor_file_tool verify <path>     reads it back and checks every value exactly
//
// The fixture is defined identically here and in tests/cross_language_roundtrip.py. If the two
// implementations of the format description disagree, one of these two directions fails.

#include "Shape.h"
#include "TensorFile.h"
#include "Tensor.h"

#include <iostream>
#include <string>
#include <vector>

namespace
{
// Deliberately awkward values: a negative, an exact zero, a fraction that is exact in binary,
// one that is not, and a large magnitude.
const std::vector<float> fixture_values = {1.5f, -2.25f, 0.0f, 3.125f, -0.1f, 100000.0f};
const veda::core::Shape fixture_shape({2, 3});
} // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: tensor_file_tool <write|verify> <path>\n";
        return 2;
    }

    const std::string command = argv[1];
    const std::string path = argv[2];

    try
    {
        if (command == "write")
        {
            veda::core::Tensor tensor{fixture_shape};
            for (size_t i = 0; i < fixture_values.size(); ++i)
            {
                tensor.data()[i] = fixture_values[i];
            }
            veda::testing::write_tensor(path, tensor);
            return 0;
        }

        if (command == "verify")
        {
            const veda::core::Tensor tensor = veda::testing::read_tensor(path);
            if (tensor.shape() != fixture_shape)
            {
                std::cerr << "shape mismatch: got " << tensor.shape().to_string() << ", expected "
                          << fixture_shape.to_string() << "\n";
                return 1;
            }
            for (size_t i = 0; i < fixture_values.size(); ++i)
            {
                if (tensor.data()[i] != fixture_values[i])
                {
                    std::cerr << "value mismatch at " << i << ": got " << tensor.data()[i]
                              << ", expected " << fixture_values[i] << "\n";
                    return 1;
                }
            }
            return 0;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << "\n";
        return 1;
    }

    std::cerr << "unknown command: " << command << "\n";
    return 2;
}
