#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>

#include "../doctest/doctest.h"
#include "Common.h"

namespace test_limits
{

bool Test1()
{
#define XTYPE(T) \
    LogInfo << #T "\t\t" \
            << std::numeric_limits<T>::min() << "\t" \
            << std::numeric_limits<T>::max() << "\t" \
            << std::numeric_limits<T>::lowest() << "\t" \
            << std::numeric_limits<T>::digits << "\t" \
            << std::numeric_limits<T>::digits10 << "\t" \
            << std::numeric_limits<T>::max_digits10 << "\t"

    LogInfo << "type\t\t" << "min()\t\t" << "max()\t\t" << "lowest()\t\t" << "digits\t\t" << "digits10\t\t" << "max_digits10"
        ;
/*
    LogInfo << "int\t"
            << std::numeric_limits<int>::min() << "\t"
            << std::numeric_limits<int>::max() << "\t"
            << std::numeric_limits<int>::lowest() << "\t"
            << std::numeric_limits<int>::digits << "\t"
            << std::numeric_limits<int>::digits10 << "\t"
            << std::numeric_limits<int>::max_digits10 << "\t";

    LogInfo << "long\t"
            << std::numeric_limits<long>::min() << "\t"
            << std::numeric_limits<long>::max() << "\t"
            << std::numeric_limits<long>::lowest() << "\t"
            << std::numeric_limits<long>::digits << "\t"
            << std::numeric_limits<long>::digits10 << "\t"
            << std::numeric_limits<long>::max_digits10 << "\t";
*/

    XTYPE(short);
    XTYPE(int);
    XTYPE(long);
    XTYPE(long long);

    XTYPE(unsigned short);
    XTYPE(unsigned int);
    XTYPE(unsigned long);
    XTYPE(unsigned long long);
    return true;
}

} // namespace test_limits

#if 0
TEST_CASE("test_limits")
{
    LogInfo << std::string(__FILE__) << ":";
    CHECK(test_limits::Test1());
}
#endif
