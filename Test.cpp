#include <string>
#include <iostream>
#include <vector>

int main()
{
    std::vector<char> v = {'\0', 0, 0};
    std::string s{v.begin(), v.end()};
    std::cout << s.size() << "\n";
    std::cout << s.length() << "\n";

    std::string str{};
    str.push_back(0);
    str.push_back(0);
    str.push_back(0);
    std::cout << str.size() << "\n";
    std::cout << str.length() << "\n";

    return 0;
}
