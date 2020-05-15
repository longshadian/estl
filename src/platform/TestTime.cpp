#include <iostream>
#include <iterator>
#include <vector>
#include <string>
#include <list>
#include <ctime>
#include <unordered_set>

#include "Common.h"

#include "../doctest/doctest.h"

namespace test_time
{

void Test1()
{
}

} // namespace test_time

//#define USE_TEST

#if defined (USE_TEST)
TEST_CASE("test_time")
{
    DPrintf("test_time");
    try {
        test_time::Test1();
    } catch (const std::exception & e) {
        printf("Error: exception: %s\n", e.what());
        CHECK(false);
    }
}
#endif

