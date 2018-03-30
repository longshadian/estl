#include <iostream>

#include <type_traits>
#include <memory>


struct A
{
    A() = default;
    ~A() = default;
    A(const A& rhs) = default;
    A& operator=(const A& rhs) = default;

    A(A&& rhs) = default;
    A& operator=(A&& rhs) = default;

    /*
    A(A&& rhs)
        : m_a(std::move(rhs.m_a))
        , m_str(std::move(rhs.m_str))
    {
        std::cout << "A move\n";
    }

    A& operator=(A&& rhs)
    {
        std::cout << "A move=\n";
        if (this != &rhs) {
            m_a = rhs.m_a;
            m_str = std::move(rhs.m_str);
        }
        return *this;
    }
    */

    int             m_a;
    std::string     m_str;
    std::unique_ptr<int> m_u;
};

struct B
{
    B() = default;
    ~B() = default;

    B(B&& rhs) = default;
    B& operator=(B&& rhs) = default;

    A a;
    int i;
};

int main()
{
    B b;
    b.i = 12;
    b.a.m_str = "afffff";
    std::cout << b.a.m_a << std::endl;
    std::cout << b.i << std::endl;

    B bb = std::move(b);
    std::cout << bb.i << std::endl;
    std::cout << bb.a.m_str << std::endl;

    std::cout <<"A is pod " << std::is_pod<A>::value << std::endl;
    std::cout <<"B is pod " << std::is_pod<B>::value << std::endl;
    std::cout << b.a.m_str << std::endl;

    A aa{};
    A aaa = aa;

    return 0;
}