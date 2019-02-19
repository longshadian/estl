
#include <atomic>
#include <thread>
#include <memory>
#include <mutex>
#include <queue>
#include <future>
#include <functional>
#include <iostream>

namespace zylib {

    class AsyncTask
    {
        struct Base
        {
            virtual ~Base() {}
            virtual void call() = 0;
        };

        template <typename F>
        struct Impl : public Base
        {
            Impl(F&& f) : m_fun(std::move(f)) {}
            virtual ~Impl() = default;
            virtual void call() { m_fun(); }

            F m_fun;
        };
    public:
        AsyncTask() = default;

        template<typename F>
        AsyncTask(F&& f) : m_impl(new Impl<F>(std::move(f)))
        {}

        AsyncTask(AsyncTask&& rhs) : m_impl(std::move(rhs.m_impl))
        {}

        AsyncTask& operator=(AsyncTask&& rhs)
        {
            if (this != &rhs) {
                m_impl = std::move(rhs.m_impl);
            }
            return *this;
        }
        AsyncTask(const AsyncTask&) = delete;
        AsyncTask& operator=(const AsyncTask&) = delete;

        operator bool() const
        {
            return m_impl != nullptr;
        }
        void operator()() { m_impl->call(); }
    private:
        std::unique_ptr<Base>   m_impl;
    };
}

template <typename T>
struct QueryResult
{
    int mysql_error_;
    T   data_;
};

template <>
struct QueryResult<void>
{
    int mysql_error_;
};

class DataBaseService
{
public:
    DataBaseService()
    {
        m_running = true;
        for (int i = 0; i != 5; ++i) {
            m_threads.push_back(std::thread(std::bind(&DataBaseService::run, this)));
        }
    }

    ~DataBaseService()
    {
        m_running = false;
        {
            std::lock_guard<std::mutex> lk(m_mtx);
            for (int i = 0; i != 5; ++i) {
                m_queue.push({});
            }
            m_cond.notify_all();
        }

        for (auto& thread : m_threads) {
            if (thread.joinable())
                thread.join();
        }
    }

    template<typename F>
    std::future<typename std::result_of<F()>::type> asyncSubmit(F f)
    {
        typedef typename std::result_of<F()>::type result_type;
        std::packaged_task<result_type()> task(std::move(f));
        auto res = task.get_future();
        std::lock_guard<std::mutex> lk(m_mtx);
        m_queue.push(std::move(task));
        m_cond.notify_all();
        return res;
    }
private:
    void run()
    {
        while (m_running) {
            zylib::AsyncTask task{};
            {
                std::unique_lock<std::mutex> lk(m_mtx);
                m_cond.wait(lk, [this] { return !m_queue.empty(); });
                std::cout << "wakeup " << std::this_thread::get_id() << "\n";
                /*
                if (!m_queue.empty()) {
                    task = std::move(m_queue.front());
                    m_queue.pop();
                }
                */
            }
            if (task) {
                task();
            }
        }
    }
private:
    mutable std::mutex          m_mtx;
    std::condition_variable     m_cond;
    std::queue<zylib::AsyncTask> m_queue;
    std::atomic<bool>           m_running;
    std::vector<std::thread>    m_threads;
};

QueryResult<int> fun()
{
    return {1, 2312};
}

QueryResult<void> funVoid()
{
    return {322};
}

int main()
{
    DataBaseService d{};
    std::this_thread::sleep_for(std::chrono::seconds(1));

    /*
    auto f = d.asyncSubmit(std::bind(&fun));
    auto f2 = d.asyncSubmit(std::bind(&funVoid));

    auto val = f.get();
    std::cout << val.mysql_error_ << " " << val.data_ << "\n";

    auto val2 = f2.get();
    std::cout << val2.mysql_error_ << "\n";
    */

    return 0;
}
