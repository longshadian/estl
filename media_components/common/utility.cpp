#include "utility.h"

#include <cstring>
#include <type_traits>
#include <memory>
#include <cstdio>
#include <sstream>

#include <sys/time.h>

namespace comm
{

std::string cat_file(const char* f)
{
    std::FILE* fp = fopen(f, "rb");
    if (!fp)
        return "";
    std::string content;
    std::array<char, 1024> buffer{};
    while (1) {
        std::size_t readn = std::fread(buffer.data(), 1, buffer.size(), fp);
        if (readn == 0)
            break;
        content.append(buffer.data(), buffer.data() + readn);
    }
    std::fclose(fp);
    return content;
}

void filter_comment(const std::string& src, std::string& dst)
{
    std::istringstream istm{src};
    std::ostringstream ostm{};
    std::string str;
    bool comment = false;
    while (!istm.eof()) {
        std::getline(istm, str);
        comment = false;
        for (const auto& c : str) {
            if (::isblank(c))
                continue;
            comment = c == '#';
            break;
        }
        if (!comment)
            ostm << str << "\n";
    }
    dst = ostm.str();
}


std::int64_t unix_time_milliseconds(const struct timeval* tv)
{
    if (tv) {
        return std::int64_t(tv->tv_sec)*1000 + std::int64_t(tv->tv_usec)/1000;
    }
    struct timeval tmp{};
    gettimeofday(&tmp, nullptr);
    tv = &tmp;
    return std::int64_t(tv->tv_sec)*1000 + std::int64_t(tv->tv_usec)/1000;
}

std::int64_t unix_time_microseconds(const struct timeval* tv)
{
    if (tv) {
        return std::int64_t(tv->tv_sec)*1000000 + std::int64_t(tv->tv_usec);
    }
    struct timeval tmp{};
    gettimeofday(&tmp, nullptr);
    tv = &tmp;
    return std::int64_t(tv->tv_sec)*1000000 + std::int64_t(tv->tv_usec);
}

int SaveFile(const char* file_name, const void* data, size_t len)
{
    FILE* f = ::fopen(file_name, "wb");
    if (!f)
        return -1;
    ::fwrite(data, 1, len, f);
    fclose(f);
    return 0;
}

} // namespace comm

