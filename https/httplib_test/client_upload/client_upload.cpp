#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <httplib.h>
using namespace httplib;
using namespace std;

static std::string cat_file(const std::string& fpath)
{
    std::ifstream istm{fpath.c_str(), std::ifstream::binary};
    istm.seekg(0, std::ios::end);
    int length = istm.tellg();
    if (!length)
        return {};
    istm.seekg(0, std::ios::beg);
    std::string buf;
    buf.resize(length);
    istm.read(&buf[0], buf.size());
    buf.resize(istm.gcount());
    return buf;
}

int main(void) 
{
    system("chcp 65001");
    Client cli("http://127.0.0.1:10086");

    std::vector<std::string> files =
    {
        "d:/x.txt",
        "C:/Users/admin/Desktop/0301bus.jpg",
        "C:/Users/admin/Desktop/111.jpg",
    };
    std::vector<std::string> contents;
    for (const auto& f : files) {
        std::string content = cat_file(f);
        assert(!content.empty());
        contents.emplace_back(std::move(content));
    }

    httplib::MultipartFormDataItems items = 
    {
        {"file1", contents[0], "x.txt", ""},
        {"file1", contents[1], "0301bus.jpg", ""},
        {"file1", contents[2], "111.jpg", ""},
    };
    auto res = cli.Post("/post", items);
    std::cout << "post res: " << res << std::endl;
    return 0;
}
