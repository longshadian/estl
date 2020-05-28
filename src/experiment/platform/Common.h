#pragma once

#include <string>
#include <chrono>

#include "console_log.h"

#define PrintDebug  CONSOLE_PRINT_DEBUG
#define PrintInfo   CONSOLE_PRINT_INFO
#define PrintWarn   CONSOLE_PRINT_WARN

#define LogDebug    CONSOLE_LOG_DEBUG
#define LogInfo     CONSOLE_LOG_INFO
#define LogWarn     CONSOLE_LOG_WARN

namespace common
{
std::string errno_to_string(int eno);
}

class PerformanceTimer
{
public:
    PerformanceTimer()
        : m_start(), m_end()
    {
        Reset();
    }

    ~PerformanceTimer() = default;
    PerformanceTimer(const PerformanceTimer&) = default;
    PerformanceTimer& operator=(const PerformanceTimer&) = default;

    void Reset()
    {
        m_start = std::chrono::steady_clock::now();
        m_end = m_start;
    }

    void Stop()
    {
        m_end = std::chrono::steady_clock::now();
    }

    template <typename T>
    T Cost() const
    {
        return std::chrono::duration_cast<T>(m_end - m_start);
    }

    std::int64_t CostMicroseconds() const
    {
        return Cost<std::chrono::microseconds>().count();
    }

    std::int64_t CostMilliseconds() const
    {
        return Cost<std::chrono::milliseconds>().count();
    }

    std::int64_t CostSeconds() const
    {
        return Cost<std::chrono::seconds>().count();
    }

    std::chrono::steady_clock::time_point m_start;
    std::chrono::steady_clock::time_point m_end;
};

