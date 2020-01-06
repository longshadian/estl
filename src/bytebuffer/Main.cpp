#include <iostream>

#include "ByteBuffer.h"
#include "ByteBufferSerializer.h"

namespace pt {

struct A
{
    int32_t m_a{};

    bool operator==(const A& rhs) const
    {
        return m_a == rhs.m_a;
    }

    META(m_a);
};

struct X
{
    int32_t m_int{0};
    std::string m_string{};
    std::vector<A> m_a{};
    std::map<int32_t, A> m_map_a{};

    bool operator==(const X& rhs) const
    {
        return m_int == rhs.m_int &&
            m_string == rhs.m_string &&
            m_a == rhs.m_a &&
            m_map_a == rhs.m_map_a
            ;
    }

    META(m_int, m_string, m_a, m_map_a);
};

}

int main()
{
    pt::X x1{};
    x1.m_int = 123;
    x1.m_string = "123523 ansdg;";
    pt::A a1{};
    a1.m_a = 188;
    x1.m_a.push_back(a1);
    x1.m_map_a.insert({1, a1});
    x1.m_map_a.insert({2, a1});
    x1.m_map_a.insert({3, a1});

    ByteBuffer bb{};
    bb << x1;

    pt::X x2{};
    bb >> x2;

    std::cout << (x1 == x2) << "\n";

    system("pause");

    return 0;
}
