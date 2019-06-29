#include "download/Utilities.h"

std::string Utilities::ToUpperCase(std::string s)
{
    // TODO
    std::string temp;
    temp.reserve(s.size());
    for (auto c : s) {
        temp.push_back(toupper(c));
    }
    return temp;
}

bool Utilities::WriteFile(const std::string& fpath, const void* data, std::size_t length)
{
    std::FILE* f = std::fopen(fpath.c_str(), "wb+");
    if (!f)
        return false;

    // TODO 能否一次写大文件？
    std::size_t remain = length;
    const char* p = reinterpret_cast<const char*>(data);
    while (remain > 0) {
        std::size_t writen = std::fwrite(p, 1, remain, f);
        remain -= writen;
        p += writen;
        std::fflush(f);
    }
    std::fclose(f);
    return true;
}
