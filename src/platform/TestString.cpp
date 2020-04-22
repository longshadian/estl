#include <iostream>
#include <string_view>
#include <vector>
#include <string>
#include <utility>

#include "Common.h"

#include "../doctest/doctest.h"

namespace test_string
{

int TestHttp()
{
    std::string s = "http://127.0.0.1:8080/";
    auto pos = s.find("://");
    std::string r1 = s.substr(0, pos);

    s = "https://127.0.0.1:8080/";
    pos = s.find("://");
    auto r2 = s.substr(0, pos);
    std::cout << r1 << " " << r2 << "\n";
    return 0;
}

std::string_view substr_by_range(std::string_view sv, std::size_t b, std::size_t e)
{
    return sv.substr(b, e-b);
}

int TestStringView()
{
    {
        std::string_view s = "http://127.0.0.1:8080";
        auto pos = s.find("://");
        CHECK(std::string_view("127.0.0.1") == substr_by_range(s, pos + 3, pos + 3 + 9));
        CHECK(std::string_view("127.0.0.1") == s.substr(7, 9));
    }

    {
        std::vector<std::pair<std::string, std::string>> vec =
        {
            {"", ""},

            {"a", "a"},
            {"a///////////", "a"},
            {"////a///////////", "a"},
            {"////a", "a"},

            {"a/b", "a/b"},
            {"a/b", "a/b"},
            {"a/b/", "a/b"},
            {"a/b////", "a/b"},
            {"//a/b////", "a/b"},
            {"//////a/b", "a/b"},

            {"a///b", "a///b"},
            {"a///b", "a///b"},
            {"a///b/", "a///b"},
            {"a///b////", "a///b"},
            {"//a///b////", "a///b"},
            {"//////a///b", "a///b"},

        };
        for (const auto& [src, v] : vec) {
            std::size_t b = src.find_first_not_of('/');
            if (b == std::string_view::npos)
                b = 0;
            std::size_t e = src.find_last_not_of('/');
            if (e == std::string_view::npos) {
                e = src.length();
            } else {
                ++e;
            }
            auto s = substr_by_range(src, b, e);
            CHECK(v == s);
        }
    }
    return 0;
}


} // namespace test_string

//#define USE_TEST

#if defined (USE_TEST)
TEST_CASE("TestString")
{
    DPrintf("TestString");
    CHECK(test_string::TestHttp() == 0);
    CHECK(test_string::TestStringView() == 0);
}
#endif
