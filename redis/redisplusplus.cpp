#include <string>
#include <vector>
#include <iostream>
#include <thread>
#include <memory>

#include <sw/redis++/redis++.h>

std::shared_ptr<sw::redis::Redis> g_redis;

int fun()
{
    sw::redis::ConnectionOptions opt{};
    opt.host = "192.168.16.231";
    opt.port = 6390;
    //opt.password = "Abcd1234";
    try {
        g_redis = std::make_shared<sw::redis::Redis>(opt);
        auto t = std::to_string(time(nullptr));
        auto ret = g_redis->ping(t);
        std::cout << "ping " << ret << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << "\n";
        return -1;
    }
}

int main()
{
    fun();
    return 0;
}

