#include "Code.h"

#include <boost/locale.hpp>

namespace zylib {

bool isUtf8(const std::string& str)
{
    return isUtf8Internal(str.data(), str.length());
}

bool isUtf8(const char* str)
{
    return isUtf8Internal(str, std::strlen(str));
}

std::basic_string<char32_t> utfToWstring(const std::string& str)
{
    try {
        return boost::locale::conv::utf_to_utf<char32_t>(str, boost::locale::conv::stop);    
    } catch (...) {
        return {};
    }
}

std::string wstringToUtf(const std::basic_string<char32_t>& str)
{
    try {
        return boost::locale::conv::utf_to_utf<char>(str, boost::locale::conv::stop);    
    } catch (...) {
        return {};
    }
}

namespace detail {
bool isUtf8Internal(const char* pos, size_t len)
{
    size_t multi_len = 0;
    for (size_t i = 0; i != len; ++i) {
        unsigned c = static_cast<unsigned char>(pos[i]);
        if (multi_len == 0) {
            if (c <= 0x7F)          //0xxx xxxx     ascii码
                multi_len = 1;
            else if (c <= 0xDF)     //110x xxxx
                multi_len = 2;
            else if (c <= 0xEF)     //1110 xxxx
                multi_len = 3;
            else if (c <= 0xF7)     //1111 0xxx
                multi_len = 4;
            else
                return false;       //不考虑4字节以后的编码
            multi_len--;
        } else {
            if ((c & 0xC0) != 0x80) //最高2bit不是10xx xxxx
                return false;
            multi_len--;
        }
    }
    if (multi_len > 0)
        return false;
    return true;
}

}
