#include <fstream>
#include <iostream>
#include <string>
#include <locale>
#include <cctype>
#include <codecvt>
#include <vector>

std::wstring toWstring(const std::string& str, const std::locale& loc = std::locale())
{
    std::vector<wchar_t> buff(str.size());
    std::use_facet<std::ctype<wchar_t>>(loc).widen(str.data(), str.data()+str.size(), buff.data());
    return std::wstring(buff.data(), buff.size());
}

std::string toString(const std::wstring& str, const std::locale& loc = std::locale())
{
    std::vector<char> buff(str.size());
    std::use_facet<std::ctype<wchar_t>>(loc).narrow(str.data(), str.data()+str.size(), '?', buff.data());
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

std::string fun(std::string path)
{
    std::fstream fs(path);
    std::string s;
    while (!fs.eof()) {
        fs >> s;
        std::cout << s << '\n';
        std::cout << s.size() << '\n';
    }
    return s;
}

void fun_utf8(std::string path)
{
    std::wfstream wfs(path);
    if (!wfs.eof()) {
        std::wstring s;
        wfs >> s;
        std::wcout << s << '\n';
        std::wcout << s.size() << '\n';
    }
}

#define  RETURN   { system("pause"); return 0; }


bool validWord(const std::string& src)
{
    auto ws = utf8ToWstring(src);

    //不是多字节编码
    if (ws.size() == src.size()) {
        for (auto c : ws) {
            if (!std::use_facet<std::ctype<wchar_t>>(std::locale()).is(
                std::ctype_base::lower | std::ctype_base::upper, c)) {
                std::cout << "input mast be all letter\n";
                return false;
            }
        }
    } else {
        //多字节，检查汉字
        if (ws.size() > 5) {
            std::cout << "input size too long:" << ws.size() << '\n';
            return false;
        }

        for (auto c : ws) {
            if (std::use_facet<std::ctype<wchar_t>>(std::locale()).is(
                std::ctype_base::lower | std::ctype_base::upper, c)) {
                std::cout << "input is lower upper\n";
                return false;
            }

            /*
            if (std::use_facet<std::ctype<wchar_t>>(std::locale()).is(
                std::ctype_base::alpha, c)) {
                std::cout << "input is alpha\n";
                return false;
            }
            */

            if (std::use_facet<std::ctype<wchar_t>>(std::locale()).is(
                std::ctype_base::space |
                std::ctype_base::blank |
                std::ctype_base::cntrl |
                std::ctype_base::punct ,c)) {
                std::cout << "input is space\n";
                return false;
            }

            if (std::use_facet<std::ctype<wchar_t>>(std::locale()).is(
                std::ctype_base::xdigit | std::ctype_base::digit, c)) {
                std::cout << "input is digit xdigit\n";
                return false;
            }

        }
    }
    return true;
}

int main()
{
    std::string path = "./res/zh.txt";
    std::string path_utf8 = "./res/zh_utf8.txt";
    //fun(path);
    auto bs = fun(path_utf8);
    std::cout << "bs:" << bs.size() << '\n';

    bs = u8"夏天天天啊";
    if (!validWord(bs)) {
        RETURN;
    } else {
        std::cout << "validWord\n"; 
        RETURN;
    }

    auto ws = utf8ToWstring(bs);
    std::cout << "ws:" << ws.size() << '\n';

    if (bs.size() == ws.size()) {
        std::cout << "is pure letter:" << bs << '\n';
        RETURN ;
    }

    int count = 0;
    for (auto c : ws) {
        if (std::use_facet<std::ctype<wchar_t>>(std::locale()).is( std::ctype_base::lower | std::ctype_base::upper, c)) {
            std::cout << "isalpha:" << toString(c) << " pos:" << count <<  '\n';
            break;
        } else {
            ++count;
        }

        std::wstring ws_temp;
        ws_temp.push_back(c);
        std::cout << wstringToUtf8(ws_temp);
    }
    std::cout << '\n';

    std::string name = "寒冷夏天天";
    std::cout << name <<'\n';
    std::cout << name.size() <<'\n';

    //fun_utf8(path_utf8);

    system("pause");
    return 0;
}