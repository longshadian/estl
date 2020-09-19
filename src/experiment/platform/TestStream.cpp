#include <iostream>
#include <iterator>
#include <vector>
#include <string>
#include <list>
#include <unordered_set>
#include <sstream>

#include "Common.h"

#include "../doctest/doctest.h"

namespace test_stream
{
static const std::string g_str =
R"(
# comment 1
{
    "a":11123,
    "b":"123 #aaa"
    #### comment222
\#### comment222
}
)";

int Test1()
{
    std::istringstream istm{g_str};
    std::ostringstream ostm{};
    std::string str;
    bool comment = false; 
    while (!istm.eof()) {
        std::getline(istm, str);
        comment = false;
        for (const auto& c : str) {
            if (std::isblank(c))
                continue;
            comment = c == '#';
            break;
        }
        if (!comment)
            ostm << str << "\n";
    }
    std::cout << ostm.str();
    return 0;
}

} // namespace test_stream

#define USE_TEST
#if defined (USE_TEST)
TEST_CASE("test_stream")
{
    LogInfo << std::string(__FILE__) << ":";
    try {
        test_stream::Test1();
    } catch (const std::exception& e) {
        PrintWarn("Error: exception: %s", e.what());
        CHECK(false);
    }
}
#endif

