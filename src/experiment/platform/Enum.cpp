#include <iostream>

#include "Common.h"

#include "../doctest/doctest.h"

namespace test_enum
{

enum class X : int
{
    A = 1,
    B = 2,
};

X& operator++(X& t)
{
    switch (t) {
    case X::A: return t = X::B;
    case X::B: return t = X::A;
    }
}

inline
std::ostream& operator<<(std::ostream& os, X x)
{
    using x_type = std::underlying_type<X>::type;
    os << static_cast<x_type>(x);
    return os;
}

/*
inline operator bool(X x)
{
    return x == X::A;
}
*/

} // namespace test_enum

//#define USE_TEST

#if defined (USE_TEST)
TEST_CASE("TestEnum")
{
    DPrintf("TestEnum");
    test_enum::X a = test_enum::X::A;
    std::cout << (bool)a << "\n";
    ++a;
    CHECK((int)a == 2);
}
#endif
