#include <iostream>
#include <chrono>
#include <random>
#include "MinMaxHeap.h"


int main()
{
    auto rand_engine = std::default_random_engine(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<int> rand_dist;

    std::vector<int> src;
    for (int i = 0; i != 1000000; ++i) {
        auto v = rand_dist(rand_engine);
        src.push_back(v);
    }

    std::vector<int> src_bk = src;
    {
        auto tb = std::chrono::system_clock::now();
        std::sort(src.begin(), src.end());
        auto te = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(te-tb).count() << std::endl;
    }

    MinMaxHeap<int> heap;
    for (auto it : src_bk) {
        heap.pushHeap(it);
    }
    {
        auto tb = std::chrono::system_clock::now();
        heap.sortHeap();
        auto te = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(te-tb).count() << std::endl;
    }

    const auto& ret = heap.getContainer();
    bool e = true;
    for (int i = 0; i != src.size(); ++i) {
        if (src[i] != ret[i]) {
            e = false;
            break;
        }
    }
    std::cout << e << std::endl;

    return 0;
}