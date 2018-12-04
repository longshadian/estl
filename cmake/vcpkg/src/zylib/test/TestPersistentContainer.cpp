#include "zylib/zylib.h"

#include <iostream>

struct A
{
    int m_a;
    double m_b;

    bool operator==(const A& rhs) const
    {
        return m_a == rhs.m_a && m_b == rhs.m_b;
    }

    bool operator!=(const A& rhs) const
    {
        return !(*this == rhs);
    }
};

void funVector()
{
    zylib::PersistentContainer<std::vector<A>> pa{};
    pa.m_data.emplace_back(A{123, 333.2});
    pa.m_data.emplace_back(A{123, 333.2});

    auto buffer = pa.serializeToBinary();

    zylib::PersistentContainer<std::vector<A>> pa_ex{};
    pa_ex.parseFromBinary(buffer);

    std::cout << (pa == pa_ex) << "\n";

    pa_ex.m_null_sentinel = 223;
    std::cout << (pa == pa_ex) << "\n";
}

void funList()
{
    zylib::PersistentContainer<std::list<A>> pa{};
    pa.m_data.emplace_back(A{123, 333.2});
    pa.m_data.emplace_back(A{123, 333.2});

    auto buffer = pa.serializeToBinary();

    zylib::PersistentContainer<std::list<A>> pa_ex{};
    pa_ex.parseFromBinary(buffer);

    std::cout << (pa == pa_ex) << "\n";
    std::cout << (pa != pa_ex) << "\n";

    pa_ex.m_null_sentinel = 223;
    std::cout << (pa == pa_ex) << "\n";
    std::cout << (pa != pa_ex) << "\n";
}

int main()
{
    funVector();
    std::cout << "===\n";
    funList();
    return 0;
}
