#include <iostream>

#include <chrono>
#include <thread>


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

int main()
{
	return 0;
}
