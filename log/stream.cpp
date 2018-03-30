
#include <iostream>
#include <string>
#include <sstream>
#include <cstdint>
#include <thread>
#include <chrono>

struct FakeLogStream
{
    FakeLogStream()
    {
    }

    std::ostringstream m_ostm;
};

template <typename T>
inline FakeLogStream& operator<<(FakeLogStream& os, const T& t)
{
    os.m_ostm << t;
    return os;
};

struct FakeLog
{
    FakeLog(int32_t lv)
        : m_level(lv)
    {
    }

    ~FakeLog()
    {
        std::string s = m_stream.m_ostm.str();
        if (!s.empty())
            std::cout << s << std::endl;
    }

    FakeLogStream& stream()
    {
        return m_stream;
    }

    int32_t m_level;
    FakeLogStream m_stream;
};

#define LOG_ERROR FakeLog(0).stream()

int main()
{
    LOG_ERROR << "ffa " << 23 << " " << 32.f;

    std::this_thread::sleep_for(std::chrono::seconds(1));
    LOG_ERROR << "xx ffa " << 23 << " " << 32.f;

    std::this_thread::sleep_for(std::chrono::seconds(1));
    LOG_ERROR << "cc ffa " << 23 << " " << 32.f;

    return 0;
}