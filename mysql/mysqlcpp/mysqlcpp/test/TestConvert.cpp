#include <iostream>

#include <cstring>
#include <string>

#include "mysqlcpp/detail/SafeString.h"
#include "mysqlcpp/detail/Convert.h"

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
    int val = 12345678;
    std::string s = std::to_string(val);

    ::mysqlcpp::detail::SafeString sf{};
    sf.resize(s.size());
    std::memcpy(sf.getPtr(), s.c_str(), s.size());
    printSafeString(sf);

    auto val_ex = mysqlcpp::detail::Convert<int>::cvt_noexcept(sf.getCString());

    std::cout << (val == val_ex) << "\n";

    return 0;
}
