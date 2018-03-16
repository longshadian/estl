#include <iostream>
#include <functional>

class B
{
public:
    B() = default;
    ~B() = default;

    std::function<void()> m_fun;
};

class A
{
public:
    A() = default;
    ~A() = default;

    void attach(B& b)
    {
        b.m_fun = std::bind(&A::fun, this);
    }

private:
    void fun()
    {
        std::cout << "call A::fun\n";
    }
};

void xx()
{

}

int main()
{
    /*
    A a;
    B b;
    //a.attach(b);
    b.m_fun();
    */

    std::function<void()> v{};
    bool b1 = static_cast<bool>(v);
    std::cout << b1 << "\n";

    v = std::bind(&xx);
    bool b2 = static_cast<bool>(v);
    std::cout << b2 << "\n";
    return 0;
}
