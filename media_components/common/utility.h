#ifndef __PAS_COMMEX_UTILITY_H
#define __PAS_COMMEX_UTILITY_H

#include <cstring>
#include <type_traits>
#include <memory>
#include <ctime>

struct timeval;

namespace comm
{

template< typename I
    , typename O
>
inline void hex_dump(I b, I e, O o)
{
    static const std::uint8_t hex[] = "0123456789abcdef";
    while (b != e) {
        *o++ = hex[*b >> 4];
        *o++ = hex[*b++ & 0xf];
    }
}

inline std::string& remove_last_char(std::string& s, char c)
{
    while (!s.empty() && s.back() == c)
        s.pop_back();
    return s;
}

template <typename T>
inline void bzero(T* t)
{
    std::memset(t, 0, sizeof(T));
    static_assert(std::is_pod<T>::value, "T must be POD!");
}

template <typename T, typename... Args>
inline std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

std::string cat_file(const char* f);
void filter_comment(const std::string& str, std::string& dst);

std::int64_t unix_time_milliseconds(const struct timeval* tv = nullptr);
std::int64_t unix_time_microseconds(const struct timeval* tv = nullptr);

inline struct tm* Localtime(const time_t* t, struct tm* output)
{
#if defined(_WIN32)
    ::localtime_s(output, t);
#else
    ::localtime_r(t, output);
#endif
    return output;
}

inline std::string Localtime_YYYYMMDD_HHMMSS(const std::time_t* t)
{
    struct tm cur_tm = { 0 };
    Localtime(t, &cur_tm);
    char buffer[64] = { 0 };

    snprintf(buffer, sizeof(buffer), "%04d%02d%02d_%02d%02d%02d"
        , cur_tm.tm_year + 1900, cur_tm.tm_mon + 1, cur_tm.tm_mday
        , cur_tm.tm_hour, cur_tm.tm_min, cur_tm.tm_sec
    );
    return std::string(buffer);
}

inline bool YYYYMMDD_HHMMSS_ToLocaltime(const std::string& str, std::time_t* t)
{
    // %04d%02d%02d_%02d%02d%02d
    if (str.size() != 15) {
        return false;
    }
    std::array<char, 16> buff{};
    const char* p = str.data();
    char* p0 = buff.data();
    std::memcpy(p0, p, 4);
    int tm_year = ::atoi(p0);

    p += 4; 
    buff.fill('\0');
    std::memcpy(p0, p, 2);
    int tm_mon = ::atoi(p0);

    p += 2;
    buff.fill('\0');
    std::memcpy(p0, p, 2);
    int tm_mday = ::atoi(p0);

    p += 3; // %02d_
    buff.fill('\0');
    std::memcpy(p0, p, 2);
    int tm_hour = ::atoi(p0);

    p += 2;
    buff.fill('\0');
    std::memcpy(p0, p, 2);
    int tm_min = ::atoi(p0);

    p += 2;
    buff.fill('\0');
    std::memcpy(p0, p, 2);
    int tm_sec = ::atoi(p0);

    struct tm cur_tm = { 0 };
    bzero(&cur_tm);
    cur_tm.tm_year = tm_year - 1900;
    cur_tm.tm_mon = tm_mon - 1;
    cur_tm.tm_mday = tm_mday;
    cur_tm.tm_hour = tm_hour;
    cur_tm.tm_min = tm_min;
    cur_tm.tm_sec = tm_sec;

    time_t ret = ::mktime(&cur_tm);
    if (ret == time_t(-1))
        return false;
    *t = ret;
    return true;
}

struct MicrosecondTimer
{
    MicrosecondTimer() : tb_() , te_() {}

    void Start() { tb_ = unix_time_microseconds(); te_ = 0; }
    void Stop() { te_ = unix_time_microseconds(); }
    std::int64_t Delta() const { return te_ - tb_; }
    float GetMilliseconds() const { return Delta()/1000.f; }

    std::int64_t tb_;
    std::int64_t te_;
};

int SaveFile(const char* file_name, const void* data, size_t len);

} // namespace comm

#endif // !__PAS_COMMEX_utility_H

