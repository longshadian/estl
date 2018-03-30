#include "Server.h"

#include <thread>

int main()
{
    boost::asio::io_service ios;
    boost::asio::io_service::work work(ios);

    zylib::Server server(ios, 9900);
    server.accept();
    std::thread t([&ios]() 
    {
        ios.run();
    });
    t.join();
    //ios.run();

    return 0;
}