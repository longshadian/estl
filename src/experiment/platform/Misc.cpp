#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "../doctest/doctest.h"
#include "Common.h"

namespace test_misc 
{

constexpr
long long operator "" _W(unsigned long long v)
{
	return v * 10000;
}

constexpr
std::chrono::seconds operator "" _s(unsigned long long v)
{
    return std::chrono::seconds{v};
}


struct Vector
{
public:
    float Sum() const
    {
        return x + y + z;
    }

    float Length() const
    {
        return x + y + z;
    }

    float x{.1};
    float y{.2};
    float z{.3};
};

struct Point : public Vector
{
public:

    float Length() = delete;
};

bool Test1()
{
    Point p;
    PrintInfo("p: %f", p.Sum());
    //PrintInfo("p: %f", p.Length());
    PrintInfo("p: %f %f %f", p.x, p.y, p.z);

    Vector v = static_cast<Vector>(p);
    PrintInfo("v: sum %f", v.Sum());
    PrintInfo("v: length %f", v.Length());
    return true;
}

} // namespace test_misc

#if 0
TEST_CASE("TestMisc")
{
    LogInfo << __FILE__;
    CHECK(test_misc::Test1());
}
#endif
