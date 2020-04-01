#pragma once

#include <memory>

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
