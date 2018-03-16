#include <boost/asio.hpp>

#include <iostream>
#include <string>
#include <thread>
#include <chrono>

int main()
{
    boost::asio::io_service io_s{};
    boost::asio::io_service::work work{ io_s };

    std::thread t{ [&io_s] 
    {
        io_s.run();
    } };

    boost::asio::ip::tcp::socket sock{ io_s };

    bool is_succ = false;
    while (!is_succ) {
        boost::system::error_code ec{};
        sock.close(ec);
        std::cout << "xxx " << ec.value() << " " << ec.message() << "\n";

        boost::asio::ip::address addr{};
        addr.from_string("127.0.0.1");
        boost::asio::ip::tcp::endpoint ep_pair{ addr, 24011 };
        sock.async_connect(ep_pair,
            [&is_succ](boost::system::error_code ec)
            {
                if (!ec) {
                    std::cout << "success\n";
                    is_succ = true;
                } else {
                    std::cout << "fail " << ec.value() << ":" << ec.message() << "\n";
                }
            });
        std::this_thread::sleep_for(std::chrono::seconds{ 2 });
    }

    t.join();
    return 0;
}
