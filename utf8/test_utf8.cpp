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

std::string splitName(std::string s, const size_t LEN)
{
    if (s.size() < LEN)
        return s;
    std::string ret;
    ret.reserve(LEN);
    try {
        std::wstring ws = boost::locale::conv::utf_to_utf<wchar_t>(s);    
        if (ws.empty())
            return "";

        std::wstring ws_temp{};
        ws_temp.reserve(LEN);
        for (auto wc : ws) {
            ws_temp.clear();
            ws_temp.push_back(wc);
            auto temp = boost::locale::conv::utf_to_utf<char>(ws_temp);    
            auto new_size = ret.size() + temp.size();
            std::cout << "temp " << temp << " " << temp.size() << " " << ws_temp.size() << "\n";
            if (new_size < LEN) {
                ret.append(temp);
            } else if (new_size == LEN) {
                ret.append(temp);
                return ret;
            } else {
                return ret;
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

        ws.resize(5);
        auto ss = boost::locale::conv::utf_to_utf<char>(ws);    
        std::cout << "split:            " << ss.size() << "\n";
        std::cout << "split size:       " << ss << "\n";
        std::cout << "split fun:        " << splitName(s) << "\n";

        auto sss = splitName(s, 18);
        std::cout << "split funEX:      " << sss << "\n";
        std::cout << "split funEX:      " << sss.size() << "\n";
    } catch (const std::exception& e) {
        std::cout << "exception:" << e.what() << "\n";
    }
    return 0;
}
