#include <iostream>

#include <unistd.h>

#include <array>
#include <cstdarg>
#include <ctime>

#include "Log.h"

struct A
{
    int v;
};

int main()
{
    zylib::ServerLogger::init("/home/cgy/work/test/log/zylog", "xx");

    int* p = nullptr;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    A a = {33333}; 
    PLOG(DEBUG) << int32_t(1);
    PLOG(DEBUG) << uint64_t(1);
    PLOG(DEBUG) << time_t(1);
    PLOG(DEBUG) << p;
    PLOG(DEBUG) << (long double)132.43123;
    PLOG(DEBUG) << "ddfsdfsdf";
    PLOG(DEBUG) << &a;

    std::this_thread::sleep_for(std::chrono::seconds(1));
    return 0;
}
