#include <iostream>

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


} // namespace test_string

//#define USE_TEST

#if defined (USE_TEST)
TEST_CASE("TestString")
{
    DPrintf("TestString");
    CHECK(test_string::TestHttp() == 0);
}
#endif
