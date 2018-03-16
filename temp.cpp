

#include <iostream>
#include <cstdint>


int main()
{
    uint64_t val = 0x7f8b00000000;
    const uint8_t* p = (const uint8_t*)&val;
    std::cout << (int)p[0] << "\n";
    std::cout << (int)p[1] << "\n";
    std::cout << (int)p[2] << "\n";
    std::cout << (int)p[3] << "\n";
    std::cout << (int)p[4] << "\n";
    std::cout << (int)p[5] << "\n";
    std::cout << (int)p[6] << "\n";
    std::cout << (int)p[7] << "\n";

    return 0;
}


