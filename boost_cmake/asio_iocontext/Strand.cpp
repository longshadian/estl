#include <iostream>
#include <thread>
#include <functional>
#include <boost/asio.hpp>

class printer
{
public:
    printer(boost::asio::io_context& io)
        : strand_(io),  // strand 成员，用于控制 handler 的执行
        timer1_(io, boost::asio::chrono::seconds(1)), // 运行两个计时器
        timer2_(io, boost::asio::chrono::seconds(1)),
        count_(0)
    {
        // 启动异步操作时，每个 handler 都绑定到 strand 对象
        // bind_executor() 返回一个新的 handler，它将自动调度其包含的 printer::print1
        // 通过将 handler 绑定到同一个 strand，保证它不会同时执行
        timer1_.async_wait(boost::asio::bind_executor(strand_,
            std::bind(&printer::print1, this)));

        timer2_.async_wait(boost::asio::bind_executor(strand_,
            std::bind(&printer::print2, this)));
    }

    ~printer()
    {
        std::cout << "Final count is " << count_ << std::endl;
    }

    void print1()
    {
        if (count_ < 10)
        {
            std::cout << "thread: " << std::this_thread::get_id() 
                <<  " Timer 1: " << count_ << std::endl;
            ++count_;

            timer1_.expires_at(timer1_.expiry() + boost::asio::chrono::seconds(1));

            timer1_.async_wait(boost::asio::bind_executor(strand_,
                std::bind(&printer::print1, this)));
        }
    }

    void print2()
    {
        if (count_ < 10)
        {
            std::cout << "thread: " << std::this_thread::get_id() << " Timer 2: " << count_ << std::endl;
            ++count_;

            timer2_.expires_at(timer2_.expiry() + boost::asio::chrono::seconds(1));

            timer2_.async_wait(boost::asio::bind_executor(strand_,
                std::bind(&printer::print2, this)));
        }
    }

private:
    boost::asio::io_context::strand strand_;
    boost::asio::steady_timer timer1_;
    boost::asio::steady_timer timer2_;
    int count_;
};

int Strand_Test()
{
    boost::asio::io_context io;
    printer p(io);
    std::thread t([&io]()
        {
            io.run();
        });
    io.run();
    t.join();
    return 0;
}
