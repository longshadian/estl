#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>

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


void Test_Set()
{
    std::set<std::pair<int, std::shared_ptr<int>>> s1;
    s1.insert({1, std::make_shared<int>(1)});
    s1.insert({1, std::make_shared<int>(2)});
    s1.insert({2, std::make_shared<int>(2)});
    s1.insert({33, std::make_shared<int>(333)});

    //auto it_end = s1.lower_bound(std::pair(1, nullptr));
    auto it_end = s1.upper_bound(std::pair(1, nullptr));
    LogInfo << int(it_end == s1.end());
    for (auto it = s1.begin(); it != it_end; ++it) {
        LogInfo << it->first << "\t" << *it->second;
    }
}

void Test_Multimap()
{
    std::multimap<char, std::shared_ptr<int>> s1;
    s1.insert({'a', std::make_shared<int>(10)});
    s1.insert({'a', std::make_shared<int>(11)});
    s1.insert({'b', std::make_shared<int>(100)});
    s1.insert({'c', std::make_shared<int>(1001)});
    s1.insert({'c', std::make_shared<int>(1002)});
    s1.insert({'c', std::make_shared<int>(1001)});
    //s1.insert({'d', std::make_shared<int>(2000)});
    s1.insert({'e', std::make_shared<int>(3000)});

    //auto it_end = s1.upper_bound('c');
    auto it_end = s1.lower_bound('c');
    CHECK(it_end != s1.end());
    for (auto it = s1.begin(); it != it_end; ++it) {
        //LogInfo << it->first << "\t" << *it->second;
    }

    {
        auto p = s1.equal_range('d');
        CHECK(p.first->first == p.second->first);
        //LogInfo << "test equal_range 1: " << p.first->first << " " << p.second->first;
        for (auto i = p.first; i != p.second; ++i) {
            //LogInfo << i->first << ": " << *i->second;
        }
    }

    {
        auto p = s1.equal_range('c');
        std::multimap<char, std::shared_ptr<int>>::iterator it = s1.end();
        //LogInfo << "test equal_range 2: " << p.first->first << " " << p.second->first;
        for (auto i = p.first; i != p.second; ++i) {
            //LogInfo << i->first << ": " << *i->second;
            if (*i->second == 1002) {
                it = i;
            }
        }
        CHECK(it != s1.end());
        auto it2 = s1.erase(it);
        CHECK(it2->first == 'c');
        CHECK(*it2->second == 1001);
        //LogInfo << "erase: " << it2->first << ": " << *it2->second;

        auto p2 = s1.equal_range('c');
        //LogInfo << "test equal_range 3: " << p2.first->first << " " << p2.second->first;
        for (auto i = p2.first; i != p2.second; ++i) {
            //LogInfo << i->first << ": " << *i->second;
        }
    }
}

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

namespace test_unordermap
{

void Test_Map()
{
    LogInfo << std::string(__FILE__);
    std::unordered_map<test_unordermap::UKey, int> m;
    test_unordermap::UKey k0 = {0 , 1, 2};
    m.insert(std::make_pair(k0, 0));

    test_unordermap::UKey k1 = {1, 2, 1};
    m.emplace(std::make_pair(k0, 1));
    CHECK(m.size()==1);
}

} // namespace test_unordermap


#define USE_TEST
#if defined (USE_TEST)
TEST_CASE("TestUnorderMap ")
{
    LogInfo << __FILE__;
    // test_unordermap::Test_Map();
    // test_unordermap::Test_Set();
    test_unordermap::Test_Multimap();
}

#endif

