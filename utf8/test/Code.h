#include <string>

namespace zylib {
/** 
 * �ж��ַ����Ƿ�utf-8��ʽ,
 * 1 bytes 0xxxxxxx 
 * 2 bytes 110xxxxx 10xxxxxx 
 * 3 bytes 1110xxxx 10xxxxxx 10xxxxxx 
 * 4 bytes 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx 
 * ע�⣺
 *   ������5,6�ֽڵ�utf-8����
 *   ����ַ�������Ϊ0,������ascii����,����true
 */
bool isUtf8(const std::string& str);
bool isUtf8(const char* str);

std::basic_string<char32_t> utfToWstring(const std::string& str);
std::string wstringToUtf(const std::basic_string<char32_t>& str);

namespace detail {

bool isUtf8Internal(const char* pos, size_t len);

}

}
