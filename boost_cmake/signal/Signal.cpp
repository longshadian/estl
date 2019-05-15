
#include <iostream>
#include <memory>
#include <string>

#include <boost/signals2.hpp>

struct X
{
    X(int i = 0)
        : m(i) 
    { };

    void operator()() const
    {
        std::cout << "val:" << m << "\n";
    }

    void print() const
    {
        std::cout << "val:" << m << "\n";
    }

    ~X()
    {
        std::cout << "X destroy:" << m << "\n";
    }

    int m;
};


int main()
{
    using signal_type = boost::signals2::signal<void()>;

    auto x = std::make_shared<X>(11);
    {
        signal_type sig{};
        //sig.connect(signal_type::slot_type(&X::print, x));
        auto c = sig.connect(std::bind(&X::print, x));

        sig();
        sig();
        c.disconnect();
        sig();

        std::cout << "sig destroy\n";
    }
    return 0;
}