#pragma once

#include <iostream>

inline
void fun(int v)
{
    {
        int v = 123;
        std::cout << v << "\n";
    }
}

