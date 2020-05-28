#include <iostream>
#include <string_view>
#include <vector>
#include <string>
#include <utility>
#include <iostream>
#include <charconv>
#include <array>

#include "Common.h"

#include "../doctest/doctest.h"

namespace test_string_to_number
{

std::string_view substr_by_range(std::string_view sv, std::size_t b, std::size_t e)
{
    return sv.substr(b, e-b);
}

int Test_stoi()
{
    try {
        std::string s = "192.100.200.55";
        std::size_t pos = 0;
        char* pend;
        LogInfo << "v1: " << std::stoi(s, &pos) << " pos: " << pos;
        LogInfo << "v2: " << std::stoi(s, &pos) << " pos: " << pos;
    } catch (const std::exception& e) {
        LogInfo << "exception: " << e.what();
    }
    return 0;
}

int Test_strtol()
{
    try {
        std::string s = "192.100.200.55.200000000000000000000000000000.111";
        const char* p = s.data();
        while (1) {
            errno = 0;
            char* pend;
            auto v = std::strtoll(p, &pend, 10);
            if (pend == p)
                break;
            LogInfo << "v: " << v << " errno: " << common::errno_to_string(errno);
            p = ++pend;
        }
    } catch (const std::exception& e) {
        LogInfo << "exception: " << e.what();
    }
    return 0;
}

int Test_from_chars()
{
    std::array<char, 10> str{ "1.234567" };
    int result;
    std::from_chars(str.data(), str.data() + 2, result);
    LogInfo << result;
    return 0;
}


} // namespace test_string

//#define USE_TEST
#if defined (USE_TEST)
TEST_CASE("TestString")
{
    LogInfo << __FILE__;
    CHECK(test_string_to_number::Test_from_chars() == 0);
}
#endif

