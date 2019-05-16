#include <iostream>

#include <cstdlib>
#include <cstring>
#include <limits>

#include "mysqlcpp/detail/Convert.h"
#include "mysqlcpp/detail/SafeString.h"

void printSafeString(const mysqlcpp::detail::SafeString& sf)
{
    std::cout << "capacity " << sf.getCapacity() << "\n";
    std::cout << "length   " << sf.getLength() << "\n";
    std::cout << "bin size " << sf.getBinary().size() << "\n";
    std::cout << "cstring  " << sf.getCString() << "\n";
    std::cout << "===================\n";
}

int main()
{
    std::string s = "12345678";

    ::mysqlcpp::detail::SafeString sf{};
    printSafeString(sf);

    sf.resize(s.size());
    std::memcpy(sf.getPtr(), s.c_str(), s.size());
    printSafeString(sf);

    {
        std::cout << "start move\n";
        ::mysqlcpp::detail::SafeString sf_ex = std::move(sf);
        printSafeString(sf);
        printSafeString(sf_ex);
    }

    {
        std::cout << "start copy\n";
        sf.clear();
        sf.resize(s.size());
        std::memcpy(sf.getPtr(), s.c_str(), s.size());
        printSafeString(sf);

        ::mysqlcpp::detail::SafeString sf_ex = sf;
        printSafeString(sf);
        printSafeString(sf_ex);
    }

    return 0;
}
