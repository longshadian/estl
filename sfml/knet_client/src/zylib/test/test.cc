#include <iostream>

#include <array>
#include <vector>
#include <thread>
#include <future>
#include <chrono>
#include <algorithm>
#include <memory>
#include <map>
#include <random>
#include <stdio.h>
#include <stdarg.h>
#include <string>
#include <list>

#include "Queue.h"
#include "ThreadPool.h"

void myprint(const char* format)
{
    while (*format) {
        if (*format == '%') {
            if (*(format + 1) == '%') {
                ++format;
            } else {
                throw "invalid format string: missing arguments";
            }
        }
        std::cout << *format++;
    }
}

template<typename Head, typename... Args>
void myprint(const char* format, const Head& value, Args... args)
{
    while (*format) {
        if (*format == '%') {
            if (*(format + 1) == '%') {
                ++format;
            } else {
                std::cout << value;
                myprint(format + 1, args...);
                return;
            }
        }
        std::cout << *format++;
    }
}

struct rankTmp
{
    rankTmp() { mId = 0; }

    int mId;
    int mFightValue;
};

bool myfun(int* p1, int* p2)
{
    return true;
}

struct PreRankPlayerGreaterFv
{
    bool operator()(const rankTmp* p1, const rankTmp* p2)
    {
        return p1->mFightValue >= p2->mFightValue;
    }
};

int main()
{
    ThreadPool pool;
    auto f = pool.submit([] { return 13; });
    std::cout << f.get() << std::endl;

    //std::count_if(rank.begin(), rank.end(), myfun);
    /*
    std::vector<rankTmp*> rank;
    for (size_t i = 0; i != 10; ++i) {
        rankTmp* p = new rankTmp();
        p->mId = i;
        rank.push_back(p);
    }
    std::sort(rank.begin(), rank.end(), myfun);
    */

    /*
    const char* f = "%d %s %s";
    myprint(f, 1, 2, 3);

    std::promise<int> p;
    auto f = p.get_future();
    auto fa = std::async(std::launch::async, [&p] {
        p.set_value(12);
        std::this_thread::sleep_for(std::chrono::seconds(3));
        std::cout << "thread end" <<std::endl;
    });

    std::cout << f.get() << std::endl;
    */
    /*
    std::thread([&p] {
        p.set_value(12);
        std::this_thread::sleep_for(std::chrono::seconds(3));
        std::cout << "thread end" <<std::endl;
    }).detach();
    */

    //myprint(1.1, 2, 3, "sf");
    //std::cout << paramCount(1, 2, "sf", 3.4);
    system("pause");
    return 0;
}
 