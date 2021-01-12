#include <functional>
#include <iostream>
#include <string>

struct X
{
    int value;

    X(int v) : value {v} {}
    ~X() = default;

#if 1
    X(const X&) = delete;
    X& operator=(const X&) = delete;
#else
    X(const X&) = default;
    X& operator=(const X&) = default;
#endif

    X(X&&) = default;
    X& operator=(X&&) = default;

    void operator()()
    {
        std::cout << "operator(): " << value << std::endl;
    }
};

void Call(X x)
{
    x();
}

template <typename T>
struct CheckT;

template <typename T>
void printType(T obj)
{
    std::cout << typeid(obj).name() << std::endl;
}

auto CreateLambda() { return [](){}; }
auto CreateLambda2() { return [](){}; }

using lambda_t = decltype([](){});

auto CreateLambda2(int i) -> decltype([](){})
{
    //if (i == 0) return []() {};
    return [](){}; 
}

auto Fun2() -> void
{

}

void Fun()
{
    /*
    X x;
    std::function<void()> f = std::bind(x);
    f();
    */
    X x{10};
    //std::function<void()> f = std::bind(&Fun1, std::move(x));
    //auto f = std::bind(&Call, std::move(x));
    //std::function<void()> f= std::bind(&Call, std::move(x));

    auto f = [x = std::move(x)]() mutable { Call(std::move(x)); };
    auto f2 = std::move(f);
    f();
    f2();
    //CheckT<decltype(f)> ct;
    printType(std::move(f));
    printType(std::move(f2));
    printType([](){});
    printType([](){});
    printType(CreateLambda());
    printType(CreateLambda());
    printType(CreateLambda2());
    printType(CreateLambda2());
    //printType(CreateLambda2(1));
}

int main()
{
    Fun();
    return 0;
}


