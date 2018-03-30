#include <iostream>

#include <thread>
#include <chrono>
#include <vector>
#include <cstring>

#include <boost/asio.hpp>

int main()
{
    boost::asio::io_service io_service;
    boost::asio::ip::tcp::socket s(io_service);
    boost::asio::ip::tcp::resolver resolver(io_service);
    boost::asio::connect(s, resolver.resolve({"127.0.0.1", "21010"}));

    uint32_t key = 1234;
    uint32_t flag = 2;
    uint32_t len = 18;

    std::vector<uint8_t> data;
    data.resize(4);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::memcpy(data.data(), &key, 4);
    auto send_n = boost::asio::write(s, boost::asio::buffer(data));
    std::cout << "send key:" << send_n << "\n";

    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::memcpy(data.data(), &flag, 4);
    send_n = boost::asio::write(s, boost::asio::buffer(data));
    std::cout << "send flag:" << send_n << "\n";

    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::memcpy(data.data(), &len, 4);
    send_n = boost::asio::write(s, boost::asio::buffer(data));
    std::cout << "send len:" << send_n << "\n";

    int n = 10;
    while (n > 0) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::vector<char> v;
        v.push_back('a');
        auto val = boost::asio::write(s, boost::asio::buffer(v));
        std::cout <<  "send :" << val << "\n";
    } 

    return 0;
}
