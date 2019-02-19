#include <iostream>

#include <string>
#include <map>
#include <vector>
#include <chrono>
#include <thread>
#include <memory>

#include <folly/futures/Future.h>
#include <folly/Memory.h>
#include <folly/Executor.h>
#include <folly/dynamic.h>
#include <folly/Baton.h>


typedef folly::FutureException eggs_t;
static eggs_t eggs("eggs");

void printThread(std::string tag = std::string())
{
    std::cout << tag << '\t' << std::this_thread::get_id() << std::endl;
}

struct Foo
{
    Foo(int id, std::string name) : m_id(id), m_name(name) {}

    int m_id{31};
    std::string m_name{"aaa"};
};


int main()
{
    folly::Promise<Foo> p;
    auto fret = p.getFuture().then([](Foo f)
    {
        std::cout << "then " << f.m_id << " " << f.m_name << std::endl;
        printThread("then");
        return "then finished";
    })
    .then([](std::string f2)
    {
        std::cout << f2 << std::endl;
        printThread("then2");
        return f2+"then2 end";
        //std::cout << "then2 " << f2.m_id << " " << f2.m_name << std::endl;
    });

    std::thread([](folly::Promise<Foo>&& prms)
    {
        printThread("thread");
        std::this_thread::sleep_for(std::chrono::seconds(3));
        std::cout << "promise " << std::endl;
        prms.setValue(Foo{22, std::string("xx")});
    }, std::move(p)).detach();

    fret.wait();
    /*
    while (!fret.isReady()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "sleep " << std::endl;
    }
    */
    printThread("main thread");
    std::cout << fret.value() << std::endl;
    //std::cout << f.value().m_id << " " << f.value().m_name << std::endl;

    //auto f = folly::makeFuture().then([] { throw eggs; });
    return 0;
}