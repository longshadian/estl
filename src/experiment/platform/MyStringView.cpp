#include <regex>
#include <string>
#include <array>
#include <iostream>

#include "Common.h"

#include "../doctest/doctest.h"

namespace test_my_string_view
{

struct my_string_view
{
    typedef char*       iterator;
    typedef const char* const_iterator;

    my_string_view()
    {

    }

    iterator begin()
    {
    }

    const_iterator begin() const
    {

    }

    char* p_;
    std::size_t len_;
};

static 
void Test1()
{

}

} // namespace test_my_string_view


#define USE_TEST

#if defined (USE_TEST)
TEST_CASE("test my string view")
{
    LogInfo << "test my string view";
    try {
        test_my_string_view::Test1();
    } catch (const std::exception& e) {
        LogWarn << e.what();
        CHECK(false);
    }
}

#endif
