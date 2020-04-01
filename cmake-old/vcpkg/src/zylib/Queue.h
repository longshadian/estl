#pragma once

#include <mutex>
#include <condition_variable>
#include <queue>
#include <memory>

namespace zylib {

template <typename T>
class ThreadSafeQueue
{
public:
    ThreadSafeQueue() = default;
    ~ThreadSafeQueue() = default;
    ThreadSafeQueue(const ThreadSafeQueue& rhs)
    {
        std::lock_guard<std::mutex> lk(rhs.m_mtx);
        m_queue= rhs.m_queue;
    }

    void push(T val)
    {
        std::lock_guard<std::mutex> lk(m_mtx);
        m_queue.push(std::move(val));
        m_cond.notify_all();
    }

    void waitAndPop(T& val)
    {
        std::unique_lock<std::mutex> lk(m_mtx);
        m_cond.wait(lk, [this] { return !m_queue.empty(); });
        val = std::move(m_queue.front());
        m_queue.pop();
    }

    std::shared_ptr<T>  waitAndPop()
    {
        std::unique_lock<std::mutex> lk(m_mtx);
        m_cond.wait(lk, [this] { return !m_queue.empty(); });
        auto res = std::make_shared(m_queue.front());
        m_queue.pop();
        return res;
    }

    bool tryPop(T& val)
    {
        std::lock_guard<std::mutex> lk(m_mtx);
        if (m_queue.empty())
            return false;
        val = std::move(m_queue.front());
        m_queue.pop();
        return true;
    }

    std::shared_ptr<T> tryPop()
    {
        std::lock_guard<std::mutex> lk(m_mtx);
        if (m_queue.empty())
            return std::shared_ptr<T>();
        auto res = std::make_shared(m_queue.front());
        m_queue.pop();
        return res;
    }

    bool empty() const
    {
        std::lock_guard<std::mutex> lk(m_mtx);
        return m_queue.empty();
    }
private:
    mutable std::mutex      m_mtx;
    std::condition_variable m_cond;
    std::queue<T>           m_queue;
};

}
