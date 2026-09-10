#include "Profile.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace veda::profile
{

namespace
{
bool enabled_ = false;
}

void Profile::record(const std::string& stage, uint64_t nanoseconds)
{
    Entry& entry = stages_[stage];
    ++entry.calls;
    entry.nanoseconds += nanoseconds;
    total_ += nanoseconds;
}

size_t Profile::calls(const std::string& stage) const
{
    const auto found = stages_.find(stage);
    return found == stages_.end() ? 0 : found->second.calls;
}

uint64_t Profile::nanoseconds(const std::string& stage) const
{
    const auto found = stages_.find(stage);
    return found == stages_.end() ? 0 : found->second.nanoseconds;
}

void Profile::reset()
{
    stages_.clear();
    total_ = 0;
}

std::string Profile::to_string() const
{
    std::vector<std::pair<std::string, Entry>> rows(stages_.begin(), stages_.end());
    std::sort(rows.begin(), rows.end(),
              [](const auto& a, const auto& b) { return a.second.nanoseconds > b.second.nanoseconds; });

    std::ostringstream out;
    out << std::left << std::setw(18) << "stage" << std::right << std::setw(8) << "calls"
        << std::setw(12) << "total ms" << std::setw(9) << "share\n";

    for (const auto& [stage, entry] : rows)
    {
        const double milliseconds = static_cast<double>(entry.nanoseconds) / 1e6;
        const double share = total_ > 0 ? 100.0 * static_cast<double>(entry.nanoseconds) /
                                              static_cast<double>(total_)
                                        : 0.0;
        out << std::left << std::setw(18) << stage << std::right << std::setw(8) << entry.calls
            << std::setw(12) << std::fixed << std::setprecision(2) << milliseconds << std::setw(8)
            << std::fixed << std::setprecision(1) << share << "%\n";
    }
    return out.str();
}

Profile& active_profile()
{
    static Profile profile;
    return profile;
}

void set_profiling_enabled(bool enabled)
{
    enabled_ = enabled;
}

bool profiling_enabled()
{
    return enabled_;
}

Scope::Scope(std::string stage)
    : stage_(std::move(stage)), started_(std::chrono::steady_clock::now()), active_(enabled_)
{
}

Scope::~Scope()
{
    if (!active_)
    {
        return;
    }
    const auto elapsed = std::chrono::steady_clock::now() - started_;
    active_profile().record(
        stage_, static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()));
}

} // namespace veda::profile
