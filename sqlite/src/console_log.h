#ifndef __CONSOLE_LOG_H
#define __CONSOLE_LOG_H

#include <cstdint>
#include <sstream>
#include <string>
#include <iostream>
#include <ctime>

#if defined(_MSC_VER)
#include <windows.h>
#else
#include <time.h>
#include <sys/time.h>
#endif

namespace console_log
{

inline
struct tm* Localtime(const time_t* t, struct tm* output)
{
#if defined(_MSC_VER)
    localtime_s(output, t);
#else
    localtime_r(t, output);
#endif
    return output;
}

inline
void Gettimeofday(struct timeval* tp)
{
#if defined(_MSC_VER)
    std::uint64_t intervals{};
    FILETIME ft{};
    GetSystemTimeAsFileTime(&ft);

    /*
     * A file time is a 64-bit value that represents the number
     * of 100-nanosecond intervals that have elapsed since
     * January 1, 1601 12:00 A.M. UTC.
     *
     * Between January 1, 1970 (Epoch) and January 1, 1601 there were
     * 134744 days,
     * 11644473600 seconds or
     * 11644473600,000,000,0 100-nanosecond intervals.
     *
     * See also MSKB Q167296.
     */

    intervals = ((std::uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    intervals -= 116444736000000000;

    tp->tv_sec = long(intervals / 10000000);
    tp->tv_usec = long((intervals % 10000000) / 10);
#else
    ::gettimeofday(tp, nullptr);
#endif // defined(_MSC_VER)
}

inline
void Localtime_YYYYMMDD_HHMMSS_F(char* buffer, size_t len)
{
    struct timeval cur_tv = { 0 };
    Gettimeofday(&cur_tv);

    struct tm cur_tm = { 0 };
    std::time_t cur_t = static_cast<std::time_t>(cur_tv.tv_sec);
    Localtime(&cur_t, &cur_tm);
    //char buffer[64] = { 0 };

#if defined(_MSC_VER)
    sprintf_s
#else
    snprintf
#endif
    (buffer, len, "%04d/%02d/%02d %02d:%02d:%02d.%06d",
        cur_tm.tm_year + 1900, cur_tm.tm_mon + 1, cur_tm.tm_mday,
        cur_tm.tm_hour, cur_tm.tm_min, cur_tm.tm_sec,
        static_cast<std::int32_t>(cur_tv.tv_usec)
    );
}

inline
std::string Localtime_YYYYMMDD_HHMMSS_F()
{
    char buffer[64] = {0};
    Localtime_YYYYMMDD_HHMMSS_F(buffer, 63);
    return std::string(buffer);
}

struct FakeLog
{
    FakeLog(std::string type, std::string file, int line)
        : ostm_(), type_(type), line_(line), file_(std::move(file))
    {
    }

    ~FakeLog()
    {
        Flush();
    }

    void Flush()
    {
        std::string s = ostm_.str();
        if (!s.empty()) {
            std::ostringstream ostm2;
            ostm2 << "[" << Localtime_YYYYMMDD_HHMMSS_F() << "]"
                << " [" << type_ << "]"
                << " [" << file_ << ":" << line_ << "] "
                << s << "\n";
            std::cout << ostm2.str();
        }
    }

    std::ostringstream& Stream()
    {
        return ostm_;
    }
    std::ostringstream ostm_;
    std::string type_;
    int line_;
    std::string file_;
};

inline
const char* file_name(const char* s)
{
    //std::string_view sv(s);
    std::string sv(s);
    auto pos = sv.rfind('/');
    if (pos == std::string::npos) {
        pos = sv.rfind('\\');
    }
    if (pos == std::string::npos)
        return s;
    return s + pos + 1;
}

} // namespace console_log

//#define CONSOLE_LOG_FILE_NAME __FILE__
#define CONSOLE_LOG_FILE_NAME console_log::file_name(__FILE__)

#define CONSOLE_LOG_TRACE   console_log::FakeLog("TRACE", CONSOLE_LOG_FILE_NAME, __LINE__).Stream()
#define CONSOLE_LOG_DEBUG   console_log::FakeLog("DEBUG", CONSOLE_LOG_FILE_NAME, __LINE__).Stream()
#define CONSOLE_LOG_INFO    console_log::FakeLog("INFO ", CONSOLE_LOG_FILE_NAME, __LINE__).Stream()
#define CONSOLE_LOG_WARN    console_log::FakeLog("WARN ", CONSOLE_LOG_FILE_NAME, __LINE__).Stream()
#define CONSOLE_LOG_CRIT    console_log::FakeLog("CRIT ", CONSOLE_LOG_FILE_NAME, __LINE__).Stream()

#define CONSOLE_PRINT_IMPL(s, fmt, ...) \
    do { \
        printf("[%s] [%s] [%s:%d] " fmt "\n", console_log::Localtime_YYYYMMDD_HHMMSS_F().c_str(), s, CONSOLE_LOG_FILE_NAME,  __LINE__, ##__VA_ARGS__); \
    } while (0)

#define CONSOLE_PRINT_TRACE(fmt, ...)  CONSOLE_PRINT_IMPL("TRACE", fmt, ##__VA_ARGS__)
#define CONSOLE_PRINT_DEBUG(fmt, ...)  CONSOLE_PRINT_IMPL("DEBUG", fmt, ##__VA_ARGS__)
#define CONSOLE_PRINT_INFO(fmt, ...)   CONSOLE_PRINT_IMPL("INFO ", fmt, ##__VA_ARGS__)
#define CONSOLE_PRINT_WARN(fmt, ...)   CONSOLE_PRINT_IMPL("WARN ", fmt, ##__VA_ARGS__)
#define CONSOLE_PRINT_CRIT(fmt, ...)   CONSOLE_PRINT_IMPL("CRIT ", fmt, ##__VA_ARGS__)

#define log_print  CONSOLE_PRINT_TRACE
#define logPrintInfo  CONSOLE_PRINT_INFO
#define logPrintWarn CONSOLE_PRINT_WARN
#define logPrintError CONSOLE_PRINT_CRIT

#define EASYLOG_INFO CONSOLE_LOG_INFO
#define EASYLOG_WARN CONSOLE_LOG_WARN

#endif
