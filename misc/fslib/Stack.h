#pragma once

#include <mutex>
#include <condition_variable>
#include <queue>
#include <memory>
#include <exception>

struct EmptyStack : std::exception 
{
    virtual const char* what() const override
    {
        return "StackEmpty!";
    }
};

template <typename T>
class ThreadSafeStack
{
public:
    ThreadSafeStack() = default;
    ~ThreadSafeStack() = default;
    ThreadSafeStack(const ThreadSafeStack& rhs)
    {
        std::lock_guard<std::mutex> lk(rhs.m_mtx);
        m_stack = rhs.m_stack;
    }

    void push(T val)
    {
        std::lock_guard<std::mutex> lk(m_mtx);
        m_stack.push(std::move(val));
    }

    std::shared_ptr<T> pop()
    {
        std::lock_guard<std::mutex> lk(m_mtx);
        if (m_stack.empty())
            throw EmptyStack();
        auto res = std::make_shared<T>(std::move(m_stack.top()));
        m_stack.pop();
        return res;
    }

    void pop(T& val)
    {
        std::lock_guard<std::mutex> lk(m_mtx);
        if (m_stack.empty())
            throw EmptyStack();
        val = std::move(m_stack.top());
        m_stack.pop();
    }

    bool empty() const
    {
        std::lock_guard<std::mutex> lk(m_mtx);
        return m_stack.empty();
    }
private:
    mutable std::mutex      m_mtx;
    std::stack<T>           m_stack;
};
