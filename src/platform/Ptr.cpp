#include <iostream>
#include <memory>
#include <type_traits>

namespace test_ptr
{

struct B
{
    explicit B(int val) : m(val) { }

    int  getVal() const { return m; }
    int getMultableVal() { return m; }

    int m;
};

using IntPtr = std::shared_ptr<B>;
using IntCPtr = std::shared_ptr<B>;

struct A
{
    A(int val) : m_p(std::make_shared<B>(val)) { }

    const IntCPtr fun() const
    {
        std::cout << "return const\n";
        return m_p;
    }

    IntPtr fun()
    {
        std::cout << "return mutable\n";
        return m_p;
    }

    IntPtr m_p;
};

} // namespace test_ptr


int TestConstPtr()
{
    test_ptr::A a{12};

    auto p = a.fun();
    p->getVal();

    const test_ptr::A& ca = a;
    auto cp = ca.fun();
    cp->getMultableVal();

    return 0;
}

