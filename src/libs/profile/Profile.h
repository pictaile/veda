#ifndef VEDA_PROFILE_H
#define VEDA_PROFILE_H

#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Where the time actually goes.
//
// Inference on a CPU is memory-bandwidth-bound rather than compute-bound: the arithmetic per token
// is about 1 GFLOP, and the memory traffic is every weight in the model — 2.4 GB at fp32, read once
// per token, none of it fitting in cache. At 20-50 GB/s that is a ceiling of 8 to 20 tokens per
// second no loop reordering can lift.
//
// So the optimisations worth doing are those that reduce bytes moved or use cycles the memory
// system is already waiting on. Which ones those are is a question only a profile answers, and this
// is the profile.
namespace veda::profile
{

class Profile
{
public:
    void record(const std::string& stage, uint64_t nanoseconds);

    // calls, total time and share, sorted by total.
    std::string to_string() const;

    uint64_t total_nanoseconds() const noexcept { return total_; }
    size_t calls(const std::string& stage) const;
    uint64_t nanoseconds(const std::string& stage) const;
    bool empty() const noexcept { return stages_.empty(); }

    void reset();

private:
    struct Entry
    {
        size_t calls = 0;
        uint64_t nanoseconds = 0;
    };

    std::unordered_map<std::string, Entry> stages_;
    uint64_t total_ = 0;
};

// The profile the instrumented model writes into, and whether it writes at all.
//
// Off by default: a normal run pays nothing. Timing at the level of stages costs tens of
// nanoseconds against milliseconds of work, which is free in practice — timing inside the inner
// loop would not be, and would change what it measures.
Profile& active_profile();
void set_profiling_enabled(bool enabled);
bool profiling_enabled();

// RAII: times a block into the active profile.
class Scope
{
public:
    explicit Scope(std::string stage);
    ~Scope();

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    std::string stage_;
    std::chrono::steady_clock::time_point started_;   // monotonic, unlike system_clock
    bool active_;
};

} // namespace veda::profile

#endif //VEDA_PROFILE_H
