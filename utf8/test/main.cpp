#include <iostream>
#include <string>
#include <locale>
#include <cctype>
#include <vector>
#include <fstream>
#include <cstdint>
#include <bitset>
//#include <codecvt>
//#include <bits/codecvt.h>

#include <boost/locale.hpp>

/** 
 * utf-8 
1 bytes 0xxxxxxx 
2 bytes 110xxxxx 10xxxxxx 
3 bytes 1110xxxx 10xxxxxx 10xxxxxx 
4 bytes 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx 
5 bytes 111110xx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 
6 bytes 1111110x 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 
不考虑4字节以后的编码
 */
bool isUtf8Detail(const char* pos, size_t len)
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

bool isUtf8(const std::string& str)
{
    return isUtf8Detail(str.data(), str.length());
}

std::string readFile(std::string path)
{
    std::string content{};
    std::fstream fs(path);
    auto a = fs.get();
    while (a != EOF) {
        content.push_back(a);
        a = fs.get();
    }
    return content;
}

void testConvert()
{
    auto d_content = readFile("./d.txt");
    //std::cout << d_content << "\n";   
    auto d_str = boost::locale::conv::to_utf<char>(d_content, "gbk");
    std::cout << d_str << "\n";
    std::cout << d_str.size() << "\n";
}

void testEmoji()
{
    std::string emoji_s = "\xF0\x9D\x8C\x86";
    for (auto c : emoji_s) {
        std::bitset<8> bs{(unsigned char)c};
        std::cout << bs << " ";
    }
    std::cout << "\n\n";
    auto emoji_wchar = boost::locale::conv::utf_to_utf<wchar_t>(emoji_s);    
    for (auto c : emoji_wchar) {
        printf("0x%08X ", c);
    }
    std::cout << "\n";
    std::cout << emoji_wchar.size() << "\n";

    auto emoji_char16 = boost::locale::conv::utf_to_utf<char16_t>(emoji_s);    
    for (auto c : emoji_char16) {
        printf("0x%08X ", c);
    }
    std::cout << "\n";
    std::cout << emoji_char16.size() << "\n";
}

std::string testCodeLen(size_t max_len, std::string content)
{
    auto wstr = boost::locale::conv::utf_to_utf<char32_t>(content);
    std::string content_ex{};
    content_ex.reserve(content.size());
    try {
        std::string str{};
        for (auto c : wstr) {
            str = boost::locale::conv::utf_to_utf<char>(std::basic_string<char32_t>{c});
            auto current_len = content_ex.size() + str.size();
            //std::cout << "current_len:" << current_len << "\n";
            if (current_len < max_len) {
                content_ex.append(str);
            } else if (current_len == max_len) {
                content_ex.append(str);
                break;
            } else {
                break;
            }
        }
        return content_ex;
    } catch (...) {
        std::cout << "throw \n";
        return {};
    }
}

void fun()
{
    //std::string content = "\x61\xE6\xB1\x89\x62\xE5\xAD\x97\x63";
    //std::string content = "\xF0\x9D\x8C\x86";
    std::string content = readFile("./c.txt");
    std::cout << "read:" << content << "\n";
    std::cout << "size:" << content.size() << "\n";
    for (auto c : content) {
        printf("0x%02X ", c);
    }
    std::cout << "\n";
    for (auto c : content) {
        printf("%d ", (int)c);
    }
    std::cout << "\n\n";

    using wc_t = char32_t;
    auto w_content = boost::locale::conv::utf_to_utf<wc_t>(content);    
    for (auto c : w_content) {
        printf("0x%08X ", c);
    }
    std::cout << "\n";
    std::cout << w_content.size() << "\n";
    std::cout << sizeof(wc_t) << "\n\n";

    std::vector<char> v;
    v.resize(w_content.size() * sizeof(wc_t));
    char* p = v.data();
    for (auto c : w_content) {
        std::memcpy(p, &c, sizeof(c));
        p += sizeof(c);
    }

    int n = 0;
    for (auto c : v) {
        printf("0x%02X ", c);
        if (++n == 4) {
            n = 0;
            printf("\n");
        }
    }
    std::cout << "\n";
    std::cout << "content isutf8:" << isUtf8(content) << "\n";

    if (w_content.size() > 4)
        w_content.resize(4);
    std::string content_x = boost::locale::conv::utf_to_utf<char>(w_content, boost::locale::conv::stop);    
    std::cout << content_x.size() << "\n";
    std::cout << (content == content_x) << "\n";
    content_x.append("..");
    std::cout << content_x << "\n";
    std::cout << isUtf8(content_x) << "\n";
}

int main()
{
    auto content = readFile("./c.txt");
    std::cout << content.size() << "\n";
    std::cout << content << "\n";
    for (size_t i = 0; i <= content.size(); ++i) {
        auto s = testCodeLen(i, content);
        printf("%d:\n", (int)i);
        printf("\t %d\n", (int)s.size());
        printf("\t %s\n", s.c_str());
    }
	return 0;
}

