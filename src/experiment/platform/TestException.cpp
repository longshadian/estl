#include <cassert>
#include <chrono>
#include <exception>
#include <iostream>
#include <thread>
#include <vector>

//#include <unistd.h>

namespace test_exception
{

struct X
{
    int64_t a;
    int64_t b;
    int64_t c;
    int64_t d;
    int64_t e;
};

void server_abort()
{
    std::abort();
}

void fxx()
{
    int n = 0;
    while (n < 3) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        n++;
    }
    std::vector<X> v;
    //v.resize(1024 * 1024 * 1024);
    v.reserve(1024 * 1024 * 1024);
}

void Test1()
{
    try {
        fxx();
    }
    catch (const std::exception & e) {
        std::string s = e.what();
        assert(0);
    }
}

int Test2()
{
    //daemon(1, 0);
    std::set_new_handler(&server_abort);

    std::thread d{ &fxx };
    d.join();
    return 0;
}

int Test3()
{
    try {
        int n = 0;
        while (n < 3) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            n++;
    }
    throw std::runtime_error("hahah");
    } catch(const std::exception& e) {
           assert(0);
    }
    return 0;
}

/*
int main()
{
    std::thread d{&fun}; 
    d.join();
    return 0;
}
*/

} // namespace test_exception

