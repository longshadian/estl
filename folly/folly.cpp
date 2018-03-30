#include <iostream>
#include <thread>
#include <string>
#include <chrono>

#include <folly/futures/Future.h>
#include <folly/futures/Promise.h>

struct ExceptionA : std::runtime_error
{
    ExceptionA(const char* a) : std::runtime_error(a) {}
    ExceptionA(std::string a) : std::runtime_error(a) {}
};

struct ExceptionB : std::runtime_error
{
    ExceptionB(const char* a) : std::runtime_error(a) {}
    ExceptionB(std::string a) : std::runtime_error(a) {}
};

struct ExceptionC : std::runtime_error
{
    ExceptionC(const char* a) : std::runtime_error(a) {}
    ExceptionC(std::string a) : std::runtime_error(a) {}
};

struct ExceptionD : std::exception
{
    ExceptionD(const char* a) : m(a) {}
    ExceptionD(std::string a) : m(a) {}

    virtual const char* what() const noexcept (true) override
    {
        return m.c_str();
    }

    std::string m;
};


int main()
{
    folly::Promise<int> p{};

    auto f = p.getFuture()
        .then([](int val)
        {
            std::cout << "then1 " << std::this_thread::get_id << "\n";
            if (val == 1)
                throw ExceptionA("eA");
            else if (val == 2)
                throw ExceptionB("eB");
            else if (val == 3)
                throw ExceptionC("eC");
            else if (val == 4)
                throw ExceptionD("eD");
            ++val;
            return ++val;
        })
        .then([](int val)
        {
            std::cout << "then2 " << std::this_thread::get_id << "\n";
            return std::to_string(val) + "_xxxx";
        })
        .onError([](ExceptionB e)
        {
            std::cout << "exception exceptionB " << e.what() << "\n";
            return std::string("exceptionB");
        })
        .onError([](ExceptionA e)
        {
            std::cout << "exception A " << e.what() << "\n";
            return folly::makeFuture<std::string>(std::runtime_error("rethrow runtime"));
            //return std::string("exceptionA");
        })
        .onError([](std::runtime_error e)
        {
            std::cout << "exception runtime_error " << e.what() << "\n";
            return std::string("runtime_error");
        })
        .onError([](folly::exception_wrapper ew)
        {
            if (ew)
                std::cout << "xxxxx ew:" << ew.what() << "\n";
            return std::string("exception_wrapper");
        });

    std::thread t([&]
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            p.setValue(1);
        });

    f.wait();

    if (f.hasValue())
        std::cout << "future value:" << f.value() << "\t" << std::this_thread::get_id() << "\n";
    else if (f.hasException()) {
        //f.value();
        std::cout << "future exception:" << std::this_thread::get_id() << "\n";
    }

    t.join();
    return 0;
}
