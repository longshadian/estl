#pragma once

#include <atomic>
#include "Queue.h"

class ThreadPool
{
    class FunWrapper
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
        FunWrapper() = default;

        template<typename F>
        FunWrapper(F&& f) : m_impl(std::make_unique<Impl<F>>(std::move(f)))
        {}

        FunWrapper(FunWrapper&& rhs) : m_impl(std::move(rhs.m_impl))
        {}

        FunWrapper& operator=(FunWrapper&& rhs)
        {
            if (this != &rhs) {
                m_impl = std::move(rhs.m_impl);
            }
            return *this;
        }
        FunWrapper(const FunWrapper&) = delete;
        FunWrapper& operator=(const FunWrapper&) = delete;

        void operator()() { m_impl->call(); }
    private:
        std::unique_ptr<Base>   m_impl;
    };
public:
    ThreadPool()
    {
        size_t thread_cout = std::thread::hardware_concurrency();
        try {
            for (size_t i = 0; i != thread_cout; ++i) {
                m_threads.push_back(std::thread(&ThreadPool::run, this));
            }

        } catch (...) {
            m_running = false;
            throw;
        }
    }

    ~ThreadPool()
    {
        m_running = false;
        for (auto& t : m_threads) {
            t.join();
        }
    }

    ThreadPool(const ThreadPool& rhs) = delete;
    ThreadPool& operator=(const ThreadPool& rhs) = delete;

    template<typename F>
    std::future<typename std::result_of<F()>::type> submit(F f)
    {
        typedef typename std::result_of<F()>::type result_type;
        std::packaged_task<result_type()> task(std::move(f));
        auto res = task.get_future();
        m_queue.push(std::move(task));
        return res;
    }
private:
    void run()
    {
        while (m_running) {
            FunWrapper task;
            if (m_queue.tryPop(task)) {
                task();
            } else {
                std::this_thread::yield();
            }
        }
    }
private:
    std::atomic<bool>           m_running = { true };
    ThreadSafeQueue<FunWrapper> m_queue;
    std::vector<std::thread>    m_threads;
};
