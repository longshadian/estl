#include <iostream>
#include <iterator>
#include <vector>
#include <string>
#include <list>
#include <unordered_set>

#include "Common.h"

#include "../doctest/doctest.h"

namespace test_iterator
{

int Test1()
{
    if (0) {
        try {
            std::vector<std::string> vec = 
            {
                "a", "b", "d", "c",
            };
            std::list<std::string> s;
            std::copy(std::begin(vec), std::end(vec), std::inserter(s, std::next(s.begin())));

            CHECK(s.size() == 4);
        } catch (const std::exception&) {
            CHECK(true);
        }
    }
    return 0;
}

} // namespace test_iterator

#define USE_TEST

#if defined (USE_TEST)
TEST_CASE("test_iterator")
{
    PrintInfo("test_iterator");
    try {
        test_iterator::Test1();
    } catch (const std::exception& e) {
        PrintWarn("Error: exception: %s", e.what());
        CHECK(false);
    }
}
#endif

