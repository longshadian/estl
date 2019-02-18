#include <iostream>
#include <type_traits>
#include "literal.h"


int main()
{
    auto v = 10_w;

    std::cout << 11_k << "\n";
    std::cout << v << "\n";

    std::cout << std::is_same<decltype(v), double>::value << "\n";

    return 0;
}