#pragma once

#include <future>

namespace zylib {


template<typename T>
class Future
{
public:
    Future() = default;
    ~Future() = default;
    Future(std::future<T>&& f)
    {
        m_future = std::move(f);
    }

    Future(Future&& rhs)
    {
        m_future = std::move(rhs.m_future);
    }

    Future& operator=(Future&& rhs)
    {
        if (this != &rhs) {
            m_future = std::move(rhs.m_future);
        }
        return *this;
    }

    T getValue() { return m_future.get(); }
    bool isValid() const { return m_future.valid(); }
    bool isReady() const
    {
        return m_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
    }
    void resetStartTime()
    {
        m_start_time = std::chrono::system_clock::now();
    }

    int64_t costSeconds() const
    {
        return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - m_start_time).count();
    }

    int64_t costMilliseconds() const
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_start_time).count();
    }

    int64_t costMicroseconds() const
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - m_start_time).count();
    }

    int64_t costNanoseconds() const
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - m_start_time).count();
    }
private:
    std::future<T> m_future{};
    std::chrono::system_clock::time_point m_start_time;
};

} // zylib
