#include <iostream>
#include <unordered_map>

namespace x {
    
struct UKey
{
    int m_idx;
    int m_a;
    int m_b;

    bool operator==(const UKey& rhs) const
    {
        return m_idx == rhs.m_idx&&
            m_a == rhs.m_a &&
            m_b == rhs.m_b;
    }
};

}

namespace std {

template <>
struct hash<x::UKey>
{
    std::size_t operator()(const x::UKey& r) const
    {
        return r.m_a + r.m_b;
    }
};

}

int main()
{
    std::unordered_map<x::UKey, int> m;
    x::UKey k0 = {0 , 1, 2};
    m.insert(std::make_pair(k0, 0));

    x::UKey k1 = {1, 2, 1};
    m.emplace(std::make_pair(k0, 1));

    std::cout << m.size() << "\n";

    return 0;
}
