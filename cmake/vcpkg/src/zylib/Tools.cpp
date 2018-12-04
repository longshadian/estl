#include "zylib/Tools.h"

#include <cstdlib>
#include <ctime>
#include <cstdarg>
#include <cstring>
#include <sstream>
#include <memory>

namespace zylib {

std::vector<std::string> stringSplit(const std::string& s, char c)
{
    std::vector<std::string> out;
    if (s.empty())
        return out;

    std::istringstream istm(s);
    std::string temp;
    while (std::getline(istm, temp, c)) {
        out.push_back(temp);
    }
    return out;
}

size_t stringReplace(std::string* str, char src, char dest)
{
    size_t t = 0;
    std::transform(str->begin(), str->end(), str->begin(),
        [&t, src, dest](char& c)
        {
            if (c == src) {
                c = dest;
                ++t;
            }
            return c;
        } );
    return t;
}

const uint8_t* ReadCString(const uint8_t* src, size_t len, std::string* out)
{
    const uint8_t* pos = nullptr;
    for (size_t i = 0; i != len; ++i) {
        if (src[i] == '\0') {
            pos = src + i;
            break;
        }
    }
    if (!pos)
        return nullptr;
    out->assign(src, pos);
    return pos + 1;
}

} // zylib
