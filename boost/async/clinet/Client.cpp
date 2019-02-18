#include "Connector.h"

#include <thread>
#include <iostream>

int main()
{
    boost::asio::io_service io_service;
    boost::asio::io_service::work work(io_service);

    zylib::Connector conn(io_service, "127.0.0.1", 9900);
    conn.start();

    std::thread t([&io_service]() {
        io_service.run();
    });

    /*
    for (int i = 0; i != 10; ++i) {
        conn.sendMsg(std::to_string(i) + ":sssss");
    }
    */

    //std::this_thread::sleep_for(std::chrono::seconds(2));
    //conn.close();

    std::this_thread::sleep_for(std::chrono::seconds(2));
    io_service.stop();
    std::this_thread::sleep_for(std::chrono::seconds(2));

    t.join();

    return 0;
}