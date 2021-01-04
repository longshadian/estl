#include <cstring>

#include <string>
#include <iostream>

#include <boost/asio.hpp>
#include <boost/asio/local/connect_pair.hpp>

boost::asio::io_context g_ioctx;

void Test()
{
    boost::asio::ip::tcp::socket sock1{g_ioctx};
    boost::asio::ip::tcp::socket sock2{g_ioctx};
    //boost::asio::local::connect_pair(sock1, sock2);
}

int main()
{
    try {
        Test();
    } catch (const std::exception& e) {
        std::cout << "exception: "<< e.what() << "\n";
        return -1;
    }

    std::cout << "main exit!\n";
    return 0;
}
