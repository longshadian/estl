
#include <string>
#include <string_view>
#include <iostream>
#include <boost/asio.hpp>

#include "func/Func.h"

int main()
{
    std::string s = "aaa";

    std::string_view sv = s;
    std::cout << sv << "\n";
    std::cout << func(3) << "\n";

    boost::asio::io_context io_ctx{};
    boost::system::error_code ec{};

    system("pause");

    return 0;
}
