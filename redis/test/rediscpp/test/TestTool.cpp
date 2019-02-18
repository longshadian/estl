#include "TestTool.h"

#include <iostream>

void pout(const std::vector<rediscpp::Buffer>& v)
{
    for (const auto& b : v) {
        std::cout << b.asString() << " ";
    }
    std::cout << std::endl;
}

void pout(const rediscpp::Buffer& v)
{
    std::cout << v.asString() << "\n";
}

void pout(const rediscpp::BufferArray& v)
{
    if (v.isBuffer())
        poutArrayCell(v);
    else {
        for (const auto& val : v) {
            poutArrayCell(val);
        }
    }
}

void poutArrayCell(const rediscpp::BufferArray& v)
{
    if (v.isBuffer()) {
        std::cout << v.getBuffer().asString() << " ";
    } else {
        std::cout << "[ " ;
        pout(v);
        std::cout << " ] ";
    }
}
