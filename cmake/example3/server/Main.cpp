
#include <string>
//#include <string_view>
#include <iostream>
//#include <boost/asio.hpp>

#include "func/Func.h"
#include "common/Common.h"

int main()
{
    std::string s = "aaa";
    std::cout << func(3) << "\n";
    std::cout << "common: " << Common::ToString(11234) << "\n";
    /*
    boost::asio::io_context io_ctx{};
    boost::system::error_code ec{};
    */

    //system("pause");

    return 0;
}
