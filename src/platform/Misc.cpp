#include <chrono>
#include <iostream>
#include <string>
#include <vector>

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


} // namespace test_misc

int TestMisc()
{
    return 0;
}

