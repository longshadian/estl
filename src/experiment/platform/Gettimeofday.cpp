#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#if defined(_MSC_VER)
    #include <windows.h>
#else
    #include <time.h>
#endif

#include "../doctest/doctest.h"
#include "Common.h"

namespace test_gettimeofday
{

static
void chrono_gettimeofday(struct timeval* out)
{
    std::uint64_t tp = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    out->tv_sec = tp / 1000;
    out->tv_usec = tp % 1000000;
}

static
void platform_gettimeofday(struct timeval* tp)
{
#if defined(_MSC_VER)
    std::uint64_t intervals{};
    FILETIME ft{};
    GetSystemTimeAsFileTime(&ft);

    /*
     * A file time is a 64-bit value that represents the number
     * of 100-nanosecond intervals that have elapsed since
     * January 1, 1601 12:00 A.M. UTC.
     *
     * Between January 1, 1970 (Epoch) and January 1, 1601 there were
     * 134744 days,
     * 11644473600 seconds or
     * 11644473600,000,000,0 100-nanosecond intervals.
     *
     * See also MSKB Q167296.
     */

    intervals = ((std::uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    intervals -= 116444736000000000;

    tp->tv_sec = long(intervals / 10000000);
    tp->tv_usec = long((intervals % 10000000) / 10);
#else
    ::gettimeofday(tp, nullptr);
#endif // defined(_MSC_VER)
}

static bool Test1()
{
    std::size_t n = 10000;
    std::vector<timeval> arr1;
    std::vector<timeval> arr2;
    arr1.resize(n);
    arr2.resize(n);

    for (std::size_t i = 0; i != n; ++i) {
        chrono_gettimeofday(&arr1[i]);
    }

    PerformanceTimer pt1;
    PerformanceTimer pt2;
    return true;
}

} // namespace test_gettimeofday

#if 1
TEST_CASE("test_gettimeofday")
{
    //LogInfo << __FILE__;
    CHECK(test_gettimeofday::Test1());
}
#endif
