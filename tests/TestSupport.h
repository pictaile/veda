#ifndef VEDA_TEST_SUPPORT_H
#define VEDA_TEST_SUPPORT_H

// A deliberately tiny test harness: no external dependency, no build system surprises.
// A test file includes this header, writes CHECK lines in main(), and returns VEDA_TEST_SUMMARY().

#include <cstdio>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#define VEDA_TEST_CAN_CHECK_ABORTS 1
#endif

namespace veda::test
{

inline int checks_run = 0;
inline int checks_failed = 0;

template <typename T>
std::string render(const T& value)
{
    std::ostringstream os;
    os << value;
    return os.str();
}

template <typename T>
std::string render(const std::vector<T>& values)
{
    std::string out = "(";
    for (size_t i = 0; i < values.size(); ++i)
    {
        if (i > 0)
        {
            out += ", ";
        }
        out += render(values[i]);
    }
    return out + ")";
}

inline void report_failure(const std::string& what, const char* file, int line)
{
    ++checks_failed;
    std::cerr << file << ":" << line << ": FAILED  " << what << "\n";
}

inline void check(bool condition, const std::string& what, const char* file, int line)
{
    ++checks_run;
    if (!condition)
    {
        report_failure(what, file, line);
    }
}

template <typename A, typename B>
void check_eq(const A& actual, const B& expected, const std::string& what, const char* file, int line)
{
    ++checks_run;
    if (!(actual == expected))
    {
        report_failure(what + "  (got " + render(actual) + ", expected " + render(expected) + ")", file, line);
    }
}

template <typename A, typename B>
void check_ne(const A& actual, const B& unexpected, const std::string& what, const char* file, int line)
{
    ++checks_run;
    if (!(actual != unexpected))
    {
        report_failure(what + "  (both are " + render(actual) + ")", file, line);
    }
}

#ifdef VEDA_TEST_CAN_CHECK_ABORTS
// Runs fn in a forked child and reports whether it died on SIGABRT — the only way to observe a
// failing assert() without taking the test process down with it.
inline bool aborts(const std::function<void()>& fn)
{
    std::cout.flush();
    std::cerr.flush();
    const pid_t pid = fork();
    if (pid == 0)
    {
        (void)std::freopen("/dev/null", "w", stderr); // the assert message is expected, not news
        fn();
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}
#endif

inline int summary(const char* suite)
{
    std::cout << suite << ": " << (checks_run - checks_failed) << "/" << checks_run << " checks passed\n";
    return checks_failed == 0 ? 0 : 1;
}

} // namespace veda::test

#define CHECK(cond) ::veda::test::check((cond), #cond, __FILE__, __LINE__)
#define CHECK_EQ(actual, expected) ::veda::test::check_eq((actual), (expected), #actual " == " #expected, __FILE__, __LINE__)
#define CHECK_NE(actual, unexpected) ::veda::test::check_ne((actual), (unexpected), #actual " != " #unexpected, __FILE__, __LINE__)

#define CHECK_THROWS_AS(expr, exception_type)                                              \
    do {                                                                                   \
        bool veda_thrown = false;                                                          \
        try { (void)(expr); }                                                              \
        catch (const exception_type&) { veda_thrown = true; }                              \
        catch (...) {}                                                                     \
        ::veda::test::check(veda_thrown, #expr " throws " #exception_type, __FILE__, __LINE__); \
    } while (false)

#ifdef VEDA_TEST_CAN_CHECK_ABORTS
#define CHECK_ABORTS(expr) \
    ::veda::test::check(::veda::test::aborts([&] { (void)(expr); }), #expr " aborts", __FILE__, __LINE__)
#else
#define CHECK_ABORTS(expr) ((void)0)
#endif

#define VEDA_TEST_SUMMARY(suite) ::veda::test::summary(suite)

#endif //VEDA_TEST_SUPPORT_H
