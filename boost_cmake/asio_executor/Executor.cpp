#include <functional>
#include <future>
#include <iostream>
#include <thread>
#include <random>
#include <vector>
#include <future>
#include <chrono>

#include <boost/asio.hpp>
#include <boost/utility.hpp>
#include <boost/thread.hpp>

struct Timer
{
    Timer() : begin_{} { }

    void Start() { begin_ = std::chrono::steady_clock::now(); }
    void Stop() { end_ = std::chrono::steady_clock::now(); }
    double Passed() const { return std::chrono::duration_cast<std::chrono::microseconds>(end_ - begin_).count() / 1.0e6; }

    std::chrono::steady_clock::time_point begin_;
    std::chrono::steady_clock::time_point end_;
};


template <typename Executor, typename F, typename... Args>
std::future<std::result_of_t<F(Args...)>> async(const Executor& ex, F&& f, Args&&... args)
{
    typedef std::result_of_t<F(Args...)> result_type;
    std::packaged_task<result_type()> work{std::forward<F>(f), std::forward<Args>(args)...};
    auto result = work.get_future();
    //boost::asio::execution::execute(ex, boost::asio::execution::blocking.always, std::move(work));
    boost::asio::execution::execute(boost::asio::require(ex, boost::asio::execution::blocking.always), std::move(work));
    //boost::asio::execution::execute(boost::asio::require(ex, boost::asio::execution::blocking.never), std::move(work));
    return result;
}

static void Test()
{
    boost::asio::static_thread_pool pool{4};
    auto ex = pool.executor();
    boost::asio::execution::execute(ex, []{
    std::cout << "thread id: " << std::this_thread::get_id() << "\n";
    });
    pool.join();
}

#if 1
static void Test2()
{
    std::default_random_engine re{};
    int v = std::uniform_int_distribution<int>(0, 100)(re);
    boost::asio::thread_pool pool{4};
    std::vector<std::future<void>> vec{};
    Timer timer{};
    timer.Start();
    for (int i = 0; i != 10; ++i) {
        auto ex = pool.executor();
        auto f = async(ex, [i]() 
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            std::cout << "thread id: " << i << " " << std::this_thread::get_id() << "\n";
        });
        vec.emplace_back(std::move(f));
    }
    for (auto& f : vec) {
        f.get();
    }
    timer.Stop();
    std::cout << "passed: " << timer.Passed() << "\n";

}
#endif

int Executor_Test()
{
    //Test();
    Test2();
    return 0;
}
