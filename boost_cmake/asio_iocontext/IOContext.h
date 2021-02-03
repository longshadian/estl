#pragma once

#include <iostream>
#include <future>
#include <thread>

#include <boost/asio.hpp>

class IOContext
{
public:
    IOContext()
        : io_ctx_{}
        , work_guard_{boost::asio::make_work_guard(io_ctx_)}
        , thd_{}
    {
    }

    ~IOContext()
    {
        work_guard_.reset();
        io_ctx_.stop();
        if (thd_.joinable()) {
            thd_.join();
        }
    }

    template<typename F>
    std::future<typename std::result_of<F()>::type> submit(F f)
    {
        typedef typename std::result_of<F()>::type result_type;
        std::packaged_task<result_type()> task(std::move(f));
        auto res = task.get_future();
        boost::asio::post(io_ctx_, [task = std::move(task)] () mutable { task(); });
        return res;
    }

    void Loop()
    {
        while (1) {
            try {
                io_ctx_.run();
                break; // run() exited normally
            } catch (const std::exception& e) {
                std::cout << "iocontext exception: " << e.what() << "\n";
            }
        }
    }

    void LoopBackground()
    {
        std::thread temp_thd([this]{ this->Loop(); });
        std::swap(this->thd_, temp_thd);
    }

    void Stop()
    {
        work_guard_.reset();
    }

    boost::asio::io_context io_ctx_;
    boost::asio::executor_work_guard<boost::asio::io_context::executor_type> work_guard_;
    std::thread thd_;
};

void IOContext_Test1();
void IOContext_Test2();
