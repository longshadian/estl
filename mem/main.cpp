#include <iostream>
#include <memory>
#include <chrono>
#include <thread>


struct A
{
    ~A()
    {
        std::cout << "~A\n";
    }

    void fp() const
    {
        std::cout << "fp\n";
    }
    std::function<void()> m_fun;  
};

int main()
{
    {
        auto a = std::make_shared<A>();
        a->m_fun = std::bind(&A::fp, a.get());

        a->m_fun();
    }
    std::this_thread::sleep_for(std::chrono::seconds(3));
    return 0;
}