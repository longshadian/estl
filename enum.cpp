#include <iostream>

namespace detail {

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

inline operator bool(X x)
{
    return x == X::A;
}

}


int main()
{
    detail::X a = detail::X::A;
    std::cout << (bool)a << "\n";
    ++a;
    std::cout << !a << "\n";
    std::cout << a << "\n";


    return 0;
}
