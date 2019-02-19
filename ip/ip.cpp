
#include <arpa/inet.h>

#include <iostream>
#include <string>
#include <cstring>

int ip2num(const char* p)
{
    return ::ntohl(::inet_addr(p));
}

std::string num2ip(int v)
{
    auto vv = ::htonl(v);
    struct in_addr a;
    std::memcpy(&a, &vv, 4);
    std::string s = inet_ntoa(a);
    return s;
}

int main()
{
    std::cout << ip2num("115.239.211.112") << "\n";
    int v = 1945097072;
    std::cout << num2ip(v) << "\n";
    return 0;
}

