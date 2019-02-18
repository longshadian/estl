
#include <iostream>
#include <string>
#include <vector>

#include "ReplayEncrypt.h"

int main()
{
    std::string key = "b6d7f334-3c95-4da6-8df";

    std::string str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    std::vector<uint8_t> out{};
    out.resize(str.size());
    replaylib::encrypt((const uint8_t*)str.data(), str.size(), (const uint8_t*)key.data(), key.size(), out.data());

    std::vector<uint8_t> outex{};
    outex.resize(str.size());
    replaylib::decrypt((const uint8_t*)out.data(), out.size(), (const uint8_t*)key.data(), key.size(), outex.data());

    std::cout << (str == std::string(outex.begin(), outex.end())) << "\n";

    for (auto v : out) {
        std::cout << (int)v << " ";
    }
    std::cout << "\n";
    for (auto v : str) {
        std::cout << (int)v << " ";
    }
    std::cout << "\n";

    return 0;
}
