
#include <cstdio>
#include <iostream>


int main()
{
    auto f = ::fopen("./temp.txt", "a+");
    if (!f) {
        std::cout << "open fail\n";
        return 0;
    }

    std::string s = "abcccc";
    auto len = ::fwrite(s.c_str(), 1, s.size(), f);
    ::fclose(f);

    auto len_ex = ::fwrite(s.c_str(), 1, s.size(), f);
    std::cout << "len: " << len << "  " << len_ex << "\n"; 

    return 0;
}
