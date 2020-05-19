#include <iostream>
#include <string_view>
#include <vector>
#include <string>
#include <utility>
#include <list>
#include <functional>
#include <condition_variable>

#include "Common.h"

#include "../doctest/doctest.h"
#include "console_log.h"

namespace test_cond
{

struct X
{
    X()
    : running_()
    , thd_()
    , mtx_()
    , cnd_()
    , buffer_()
    {

    }

    ~X()
    {
        running_ = false;
        if (thd_.joinable())
            thd_.join();
    }

    void init()
    {
        running_ = true;
        std::thread tmp(std::bind(&X::run, this));
        thd_ = std::move(tmp);
    }

    void post(std::string s)
    {
        std::lock_guard<std::mutex> lk{mtx_};
        buffer_.emplace_back(std::move(s));
        cnd_.notify_all();
    }

    void run()
    {
        while (running_) {
            std::string s;
            {
                std::unique_lock<std::mutex> lk{mtx_};
                auto rv = cnd_.wait_for(lk, std::chrono::seconds(100), [this](){ return !buffer_.empty(); });
                if (buffer_.empty()) {
                    std::cout << "buffer empty\n";
                    continue;
                }
                s = std::move(buffer_.front());
                buffer_.pop_front();
                std::cout << "buffer: " << s << "\n";
            }
        }
    }

    bool                    running_;
    std::thread             thd_;
    std::mutex              mtx_;
    std::condition_variable cnd_;
    std::list<std::string>  buffer_;
};


static int Test1()
{
    X x{};
    x.init();

    int n = 0;
    while (1) {
        std::this_thread::sleep_for(std::chrono::seconds{2});
        x.post(std::to_string(++n));
        if (n == 5) {
            break;
        }
    }

    x.cnd_.notify_all();
    return 0;
}

} // namespace test_cond

#if 0
TEST_CASE("TestCondition")
{
    DPrintf("TestCondition");
    CONSOLE_PRINT_INFO("TestCondition");
    CHECK(test_cond::Test1() == 0);
}
#endif

