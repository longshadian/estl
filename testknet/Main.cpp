#include "Client.h"
#include "Connector.h"


int main()
{
    boost::asio::io_context io_ctx;
    boost::asio::io_context::work work(io_ctx);

    Connector conn(io_ctx);
    conn.Start("192.168.0.242", 8087);

    std::thread t([&io_ctx]() {
        io_ctx.run();
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));
    io_ctx.stop();
    std::this_thread::sleep_for(std::chrono::seconds(2));

    t.join();

    system("pause");
    return 0;
}
