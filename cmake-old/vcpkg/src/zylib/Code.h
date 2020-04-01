#pragma once

#include <string>

namespace zylib {
/** 
 * 判断字符串是否utf-8格式,
 * 1 bytes 0xxxxxxx 
 * 2 bytes 110xxxxx 10xxxxxx 
 * 3 bytes 1110xxxx 10xxxxxx 10xxxxxx 
 * 4 bytes 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx 
 * 注意：
 *   不考虑5,6字节的utf-8编码
 *   如果字符串长度为0,或者是ascii编码,返回true
 */
bool isUtf8(const std::string& str);
bool isUtf8(const char* str);

std::basic_string<char32_t> utfToWstring(const std::string& str);
std::string wstringToUtf(const std::basic_string<char32_t>& str);

/// 截取返回的字符串确保是合法utf8字符串
/// 如果content不是utf8编码，结果未定义
std::string utf8SplitByte(const std::string& content, size_t max_byte);

namespace detail {

bool isUtf8Internal(const char* pos, size_t len);

}

}
