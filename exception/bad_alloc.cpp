#include <iostream>
#include <chrono>
#include <thread>
#include <exception>
#include <cassert>
#include <vector>
#include <unistd.h>

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

void fun()
{
    try {
        fxx();
       } catch(const std::exception& e) {
           std::string s = e.what();
           assert(0);
       }
}

int main()
{
    daemon(1, 0);
    std::set_new_handler(&server_abort);

    std::thread d{&fun}; 
    d.join();
    return 0;
}
