

#include <thread>
#include <string>
#include <vector>
#include <iostream>

#include <boost/asio.hpp>

class EventManager
{
public:
    EventManager(){}
    ~EventManager() { }


};

void threadRun(boost::asio::io_service* ios)
{
    try {
        ios->run();
    } catch (const std::exception& e) {
        std::cout << "thread run exception: " << e.what() << "\n";
        return ;
    }

    std::cout << "ios exit\n";
}

void doRead(boost::asio::local::stream_protocol::socket* sock)
{
    auto buf = std::make_shared<std::array<char, 10>>();
    sock->async_read_some(boost::asio::buffer(*buf)
        , [sock, buf](boost::system::error_code ec, size_t len) {
            if (ec) {
                std::cout << "ec: " << ec.message() << "\n";
                return;
            }
            for (size_t i = 0; i != len; ++i) {
                std::cout << "read: " << buf->at(i) << "\n";
            }

            doRead(sock);
        });
}

int main()
{
    boost::asio::io_service ios{};

    size_t cnt = 1;
    auto tbefor = std::chrono::system_clock::now();
    for (size_t i = 0; i != cnt; ++i) {
        try {

            boost::asio::local::stream_protocol::socket sock_read{ios};
            boost::asio::local::stream_protocol::socket sock_write{ios};

            boost::asio::local::connect_pair(sock_read, sock_write);

            sock_read.shutdown(boost::asio::socket_base::shutdown_both);
            sock_read.close();

            sock_write.shutdown(boost::asio::socket_base::shutdown_both);
            sock_write.close();

        } catch (const std::exception& e) {
            std::cout << "exception: " << e.what() << "\n";
            return 0;
        }
    }

    auto tend = std::chrono::system_clock::now();

    std::cout << "create cnt: " << cnt 
        << "    cost: " << std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbefor).count() << "ms\n";

    boost::asio::io_service::work work{ios};
    std::thread t{std::bind(&threadRun, &ios)};

    t.join();

    /*
    doRead(&sock_read);

    while (true) {
        std::string s = std::to_string(std::time(nullptr));
        sock_write.write_some(boost::asio::buffer(s));
        std::this_thread::sleep_for(std::chrono::seconds{1});
    }

    t.join();
    */
    return 0;
}
