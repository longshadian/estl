#include <iostream>
#include <random>

#include <map>

std::default_random_engine g_engine{static_cast<unsigned int>(time(nullptr))};

constexpr
int64_t operator "" _y(unsigned long long v)
{
    return v * 100000000;
}

template <typename T>
inline
T rand(T closed_begin, T closed_end)
{
    return std::uniform_int_distribution<T>(closed_begin, closed_end)(g_engine);
}


inline
int64_t rand64()
{
    return std::uniform_int_distribution<int64_t>()(g_engine);
}

int main()
{
    std::map<int, int> m;
    for (int i = 0; i != 100000000; ++i) {
        auto v = rand(0_y, 10_y);
        m[v/1_y]++;
    }

    for (auto v : m) {
        std::cout << v.first << "\t" << v.second << "\n";
    }

    return 0;
}