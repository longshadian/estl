#include <iostream>
#include <unordered_map>

#include "../doctest/doctest.h"
#include "Common.h"

namespace test_unordermap 
{
    
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

} // namespace test_unordermap

namespace std 
{

template <>
struct hash<test_unordermap::UKey>
{
    std::size_t operator()(const test_unordermap::UKey& r) const
    {
        return r.m_a + r.m_b;
    }
};

} // namespace std

#define USE_TEST

#if defined (USE_TEST)
TEST_CASE("TestUnorderMap ")
{
    PrintInfo("TestUnorderMap");
    std::unordered_map<test_unordermap::UKey, int> m;
    test_unordermap::UKey k0 = {0 , 1, 2};
    m.insert(std::make_pair(k0, 0));

    test_unordermap::UKey k1 = {1, 2, 1};
    m.emplace(std::make_pair(k0, 1));
    CHECK(m.size()==1);
}

#endif

