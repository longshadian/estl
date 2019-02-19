#include <fstream>
#include <iostream>
#include <string>
#include <boost/locale.hpp>

std::string readFile(std::string path)
{
    std::string content{};
    std::ifstream fs(path);
    auto a = fs.get();
    while (a != EOF) {
        content.push_back(a);
        a = fs.get();
    }
    return content;
}

std::string splitName(std::string s)
{
    const size_t MAX_LEN = 8;
    std::locale loc{};
    std::string ret;
    for (auto c : s) {
        if (std::isgraph(c, loc)) {
            ret.push_back(c);
            if (ret.size() == 8)
                return ret;
        } else {
            break;
        }
    }

    try {
        std::wstring ws = boost::locale::conv::utf_to_utf<wchar_t>(s);    
        if (ws.size() > 5)
            ws.resize(5);
        ret = boost::locale::conv::utf_to_utf<char>(ws);    
    } catch (const std::exception& e) {
        ret.clear();
    }
    return ret;
}

std::string splitName(std::string src, const size_t LEN)
{
    if (src.size() < LEN)
        return src;
    std::string ret;
    ret.reserve(LEN);
    try {
        std::wstring ws = boost::locale::conv::utf_to_utf<wchar_t>(src);    
        if (ws.empty())
            return "";

        for (auto wc : ws) {
            auto str = boost::locale::conv::utf_to_utf<char>(std::wstring{wc});    
            auto new_size = ret.size() + str.size();
            std::cout << "str " << str << " " << str.size() << " " << "\n";
            if (new_size < LEN) {
                ret.append(str);
            } else if (new_size == LEN) {
                ret.append(str);
                break;
            } else {
                break;
            }
        }
        return ret;
    } catch (const std::exception& e) {
        return "";
    }
}

//按照字符个数截断
std::string splitCharacter(std::string src, const size_t LEN)
{
    if (src.size() <= LEN)
        return src;
    std::string ret;
    ret.reserve(LEN);
    try {
        //宽字符的字节数小于
        std::wstring wsrc = boost::locale::conv::utf_to_utf<wchar_t>(src);    
        if (wsrc.size() < LEN)
            return src;

        for (auto wc : wsrc) {
            auto str = boost::locale::conv::utf_to_utf<char>(std::wstring{wc});    
            auto new_size = ret.size() + str.size();
            std::cout << "str " << str << " " << str.size() << " " << "\n";
            if (new_size < LEN) {
                ret.append(str);
            } else if (new_size == LEN) {
                ret.append(str);
                break;
            } else {
                break;
            }
        }
        return ret;
    } catch (const std::exception& e) {
        return "";
    }
}



int main()
{
    auto s = readFile("./b.txt");
    try {
        std::wstring ws = boost::locale::conv::utf_to_utf<wchar_t>(s);    
        std::cout << "string:           " << s << "\n";
        std::cout << "string size():    " << s.size() << "\n";
        std::cout << "wstring:          " << ws.size() << "\n";

        std::cout << "\n\nsplit:5\n";
        ws.resize(6);
        auto ss = boost::locale::conv::utf_to_utf<char>(ws);    
        std::cout << "split:            " << ss.size() << "\n";
        std::cout << "split size:       " << ss << "\n";
        std::cout << "split fun:        " << splitName(s) << "\n";

        std::cout << "\n\nsplit:18\n";
        auto sss = splitName(s, 18);
        std::cout << "split funEX:      " << sss << "\n";
        std::cout << "split funEX:      " << sss.size() << "\n";
        std::cout << sizeof(wchar_t) << "\n";
    } catch (const std::exception& e) {
        std::cout << "exception:" << e.what() << "\n";
    }
    return 0;
}
