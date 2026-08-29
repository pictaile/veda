#ifndef VEDA_STORAGE_H
#define VEDA_STORAGE_H

#include <cstddef>
#include <vector>

namespace veda::core
{

// A flat, one-dimensional block of floats. Owns the memory; knows nothing about shape.
//
// Non-copyable on purpose: weights are gigabytes, and an accidental copy should be a compile
// error rather than a slow run. Tensors share it through std::shared_ptr<Storage>.
class Storage
{
public:
    // Zero-initialised: uninitialised floats become NaNs that propagate silently through softmax.
    explicit Storage(size_t n_elements) : data_(n_elements, 0.0f) {}

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    Storage(Storage&&) = default;
    Storage& operator=(Storage&&) = default;

    float* data() noexcept { return data_.data(); }
    const float* data() const noexcept { return data_.data(); }

    // In elements, never bytes.
    size_t size() const noexcept { return data_.size(); }

private:
    std::vector<float> data_;
};

} // namespace veda::core

#endif //VEDA_STORAGE_H
