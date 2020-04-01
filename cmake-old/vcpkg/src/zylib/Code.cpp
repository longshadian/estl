#include "zylib/Code.h"

#include <cstring>
#include <boost/locale.hpp>

namespace zylib {

bool isUtf8(const std::string& str)
{
    return detail::isUtf8Internal(str.data(), str.length());
}

bool isUtf8(const char* str)
{
    return detail::isUtf8Internal(str, std::strlen(str));
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

std::string utf8SplitByte(const std::string& content, size_t max_byte)
{
    if (content.size() <= max_byte)
        return content;
    auto wstr = utfToWstring(content);
    std::string sub_content{};
    sub_content.reserve(content.size());
    try {
        std::string str{};
        for (auto c : wstr) {
            str = wstringToUtf(std::basic_string<char32_t>{c});
            auto current_byte = sub_content.size() + str.size();
            if (current_byte < max_byte) {
                sub_content.append(str);
            } else if (current_byte == max_byte) {
                sub_content.append(str);
                break;
            } else {
                break;
            }
        }
        return sub_content;
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
            if ((c & 0x80) == 0)                // 1000 0000 以下是ascii码
                multi_len = 1;
            else if ((c & 0xE0) == 0xC0)        // 110x xxxx   0xE0是11100000  0xC0是11000000  
                multi_len = 2;
            else if ((c & 0xF0) == 0xE0)        // 1110 xxxx   0xF0是11110000  0xE0是11100000  
                multi_len = 3;
            else if ((c & 0xF8) == 0xF0)        // 1111 0xxx   0xF8是11111000
                multi_len = 4;
            else
                return false;                   // 不考虑4字节以后的编码
            /*
            else if ((c & 0xFC) == 0xF8)
                count = 5;
            else if ((c & 0xFE) == 0xFC)
                count = 6;
            */
            multi_len--;
        } else {
            if ((c & 0xC0) != 0x80) // 最高2bit不是10xx xxxx
                return false;
            multi_len--;
        }
    }
    if (multi_len > 0)
        return false;
    return true;
}

}
}
