#include "TestNet.h"

#include <thread>
#include <chrono>
#include <iostream>

#include "net/Udp.h"

namespace testnet
{

void TestUdpServer()
{
    std::thread t{ 
        []()
        {
            boost::asio::io_context ioc;
            UdpServer upd_server{ioc, 10086};
            ioc.run();
        }
    };
    t.detach();
}

void OnRead(std::shared_ptr<std::array<char, 1024>> buf, const boost::system::error_code& error,
    std::size_t bytes_transferred)
{
    if (error) {
        std::cout << "Error: OnRead " << error.message() << "\n";
    } else {
        std::string s{ buf->begin(), buf->begin() + bytes_transferred };
        std::cout << "buf: OnRead " << s << "\n";
    }
}

void TestUdpClinet()
{
    using namespace boost::asio::ip;

    std::thread t{ 
        []()
        {
            int n = 0;
            UdpClient client{8080, "127.0.0.1", 10086};
            while (true) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                auto str = std::to_string(++n);
                auto n = client.SendPacket(str.c_str(), str.size());
                //printf("client send size: %d\n", n);
            }
        } 
    };
    t.detach();

    boost::asio::io_context ioc{};
    udp::socket sock{ioc, udp::endpoint(udp::v4(), 10086)};
    auto buf = std::make_shared<std::array<char, 1024>>();
    udp::endpoint remote_endpoint{};
    while (true) {
        ioc.restart();
        sock.async_receive_from(boost::asio::buffer(*buf), remote_endpoint
            , std::bind(&OnRead, buf, std::placeholders::_1, std::placeholders::_2));
        ioc.run_one_for(std::chrono::milliseconds{ 1 });
        //std::cout << "run_one_for\n";
    }
}

void TestNet()
{
    TestUdpServer();

    int32_t n = 0;
    while (false) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        UdpClient client{"127.0.0.1", 10086};
        auto str = std::to_string(++n);
        auto n = client.SendPacket(str.c_str(), str.size());
        printf("client size: %d\n", n);

        UdpClient client1{10089, "127.0.0.1", 10086};
        auto str1 = std::to_string(++n);
        auto n1 = client1.SendPacket(str1.c_str(), str1.size());
        printf("client1 size: %d\n", n1);
    }

    UdpClient client{"127.0.0.1", 10086};
    while (true) {
        //std::this_thread::sleep_for(std::chrono::seconds(2));
        auto tbegin = std::time(nullptr);
        std::array<char, 1024> buffer;
        auto n = client.GetPacketBlocking(buffer.data(), buffer.size(), 2000);
        auto tend = std::time(nullptr);
        printf("client1 size: %d %d\n", n, tend - tbegin);
    }
}

}
