#include <chrono>
#include <iostream>
#include <vector>

#include "zylib/Random.h"

int main()
{
    zylib::DefaultRandom r{};
    int n = 0;
    std::vector<int16_t> v = {1,2,3,4,5,6,7,8,9};
    while (n != 10) {
        std::cout << r.rand<int64_t>() << "\t" << r.rand<int32_t>(-100000, 100000) << "\n";
        ++n;
    }
    r.shuffle(v.begin(), v.end());
    for (auto val : v) {
        std::cout << val << " ";
    }
    std::cout << "\n";
    return 0;
}
