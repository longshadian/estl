#include <iostream>
#include <iterator>
#include <vector>
#include <string>
#include <unordered_set>

#include "Common.h"

#include "../doctest/doctest.h"

namespace test_iterator
{

int Test1()
{
    {
        std::vector<std::string> vec = 
        {
            "a", "b", "a", "c",
        };
        std::unordered_set<std::string> s;
        std::copy(std::begin(vec), std::end(vec), std::inserter(s, std::next(s.begin())));

        CHECK(s.size() == 3);
    }

    {
        std::vector<std::string> vec = 
        {
            "a", "b", "d", "c",
        };
        std::unordered_set<std::string> s;
        std::copy(std::begin(vec), std::end(vec), std::inserter(s, std::next(s.begin())));

        CHECK(s.size() == 4);
    }
    return 0;
}

} // namespace test_iterator

#define USE_TEST

#if defined (USE_TEST)
TEST_CASE("test_iterator")
{
    DPrintf("test_iterator");
    try {
        test_iterator::Test1();
    } catch (const std::exception& e) {
        printf("Error: exception: %s\n", e.what());
        CHECK(false);
    }
}
#endif

