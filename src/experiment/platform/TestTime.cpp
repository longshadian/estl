#include <iostream>
#include <iterator>
#include <vector>
#include <string>
#include <list>
#include <ctime>
#include <array>
#include <unordered_set>

#include "Common.h"

#include "../doctest/doctest.h"

namespace test_time
{


    template <typename T>
    inline void bzero(T* t)
    {
        std::memset(t, 0, sizeof(T));
        static_assert(std::is_pod<T>::value, "T must be POD!");
    }



    inline struct tm* Localtime(const time_t* t, struct tm* output)
    {
#if defined(_WIN32)
        ::localtime_s(output, t);
#else
        ::localtime_r(t, output);
#endif
        return output;
    }


    inline std::string YYYYMMDD_HHMMSS_ToLocaltime(const std::time_t* t)
    {
        struct tm cur_tm = { 0 };
        Localtime(t, &cur_tm);
        char buffer[64] = { 0 };

        snprintf(buffer, sizeof(buffer), "%04d%02d%02d_%02d%02d%02d"
            , cur_tm.tm_year + 1900, cur_tm.tm_mon + 1, cur_tm.tm_mday
            , cur_tm.tm_hour, cur_tm.tm_min, cur_tm.tm_sec
        );
        return std::string(buffer);
    }

    inline bool Localtime_YYYYMMDD_HHMMSS_To(const std::string& str, std::time_t* t)
    {
        // %04d%02d%02d_%02d%02d%02d
        if (str.size() != 15) {
            return false;
        }
        std::array<char, 16> buff{};
        const char* p = str.data();
        char* p0 = buff.data();
        std::memcpy(p0, p, 4);
        int tm_year = ::atoi(p0);

        p += 4;
        buff.fill('\0');
        std::memcpy(p0, p, 2);
        int tm_mon = ::atoi(p0);

        p += 2;
        buff.fill('\0');
        std::memcpy(p0, p, 2);
        int tm_mday = ::atoi(p0);

        p += 3; // %02d_
        buff.fill('\0');
        std::memcpy(p0, p, 2);
        int tm_hour = ::atoi(p0);

        p += 2;
        buff.fill('\0');
        std::memcpy(p0, p, 2);
        int tm_min = ::atoi(p0);

        p += 2;
        buff.fill('\0');
        std::memcpy(p0, p, 2);
        int tm_sec = ::atoi(p0);

        struct tm cur_tm = { 0 };
        bzero(&cur_tm);
        cur_tm.tm_year = tm_year - 1900;
        cur_tm.tm_mon = tm_mon - 1;
        cur_tm.tm_mday = tm_mday;
        cur_tm.tm_hour = tm_hour;
        cur_tm.tm_min = tm_min;
        cur_tm.tm_sec = tm_sec;

        time_t ret = ::mktime(&cur_tm);
        if (ret == time_t(-1))
            return false;
        *t = ret;
        return true;
    }


void Test1()
{
    auto tnow = std::time(nullptr);
    auto str = YYYYMMDD_HHMMSS_ToLocaltime(&tnow);
    LogInfo << tnow;
    LogInfo << str;

    time_t xnow = 0;
    Localtime_YYYYMMDD_HHMMSS_To(str, &xnow);
    LogInfo << xnow;
}

} // namespace test_time

#define USE_TEST

#if defined (USE_TEST)
TEST_CASE("test_time")
{
    LogInfo << "test_time";
    try {
        test_time::Test1();
    } catch (const std::exception & e) {
        printf("Error: exception: %s\n", e.what());
        CHECK(false);
    }
}
#endif

