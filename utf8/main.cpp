#include <iostream>
#include <string>
#include <locale>
#include <cctype>
#include <vector>
#include <fstream>
#include <cstdint>
//#include <bits/codecvt.h>

std::wstring toWstring(const std::string& str, const std::locale& loc = std::locale())
{
    std::vector<wchar_t> buff(str.size());
    std::use_facet<std::ctype<wchar_t>>(loc).widen(str.data(), str.data() + str.size(), buff.data());
    return std::wstring(buff.data(), buff.size());
}

std::string toString(const std::wstring& str, const std::locale& loc = std::locale())
{
    std::vector<char> buff(str.size());
    std::use_facet<std::ctype<wchar_t>>(loc).narrow(str.data(), str.data() + str.size(), '?', buff.data());
    return std::string(buff.data(), buff.size());
}

std::string toString(wchar_t c, const std::locale& loc = std::locale())
{
    std::wstring temp;
    temp.push_back(c);
    return toString(temp, loc);
}

std::wstring toWstring(char c, const std::locale& loc = std::locale())
{
    std::string temp;
    temp.push_back(c);
    return toWstring(temp, loc);
}

/*
std::wstring utf8ToWstring(const std::string& str)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>> myconv;
    return myconv.from_bytes(str);
}

std::string wstringToUtf8(const std::wstring str)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>> myconv;
    return myconv.to_bytes(str);
}
*/

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

int main()
{
    auto content = readFile("./b.txt");
    std::cout << "read:" << content << "\n";
    std::cout << "size:" << content.size() << "\n";

    auto w_content = toWstring(content);
    std::wcout << "read:" << w_content << "\n";
    std::wcout << "size:" << w_content.size() << "\n";
	return 0;
}
