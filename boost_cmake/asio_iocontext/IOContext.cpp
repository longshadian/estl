#include <cstring>

#include <string>
#include <iostream>

#include <boost/asio.hpp>

#include "IOContext.h"

boost::asio::io_context g_ioctx{};
IOContext g_ctx{};

void EventLoop()
{
    boost::asio::executor_work_guard<boost::asio::io_context::executor_type>
        work_guard = boost::asio::make_work_guard(g_ioctx);
    g_ioctx.run();
}

void Test1()
{
    boost::asio::post([]{});
    EventLoop();
}

void Test2()
{
    g_ctx.LoopBackground();

    auto f1 = g_ctx.submit([]{
        int n = 0;
        while (n < 10) {
            ++n;
            std::this_thread::sleep_for(std::chrono::seconds{1});
            std::cout << "n: " << n << "\n";
        }
    });

    auto f2 = g_ctx.submit([]() -> std::string { return "f2 10"; });
    std::cout << "wait for f2: " << std::endl;
    std::cout << "f2: " << f2.get() << "\n";
    g_ctx.Stop();
}

int main()
{
    try {
        Test2();
    } catch (const std::exception& e) {
        std::cout << "exception: "<< e.what() << "\n";
        return -1;
    }

    std::cout << "main exit!\n";
    return 0;
}
