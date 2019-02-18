#include "RedisCpp.h"

#include <cstdio>
#include <string>
#include <chrono>
#include <iostream>
#include "TestTool.h"

std::string ip = "127.0.0.1";
int port = 6379;

using namespace rediscpp;

ContextGuard g_context;

bool test()
{
    try {
        Buffer key{"a"};
        DEL(g_context, key);

        RedisList redis{g_context};
        TEST(redis.RPUSH(key, Buffer("b")) == 1);
        TEST(redis.RPUSH(key, Buffer(123)) == 2);
        TEST(redis.LPUSH(key, Buffer(12.98)) == 3);
        TEST(redis.LLEN(key) == 3);

        auto ret1 = redis.LRANGE(key, 0, -1);
        pout(ret1);

        auto ret2 = redis.RPOP(key);
        pout(ret2);

        auto ret3 = redis.LRANGE(key, 0, -1);
        pout(ret3);

        auto ret4 = redis.LPOP(key);
        pout(ret4);

        auto ret5 = redis.LRANGE(key, 0, -1);
        pout(ret5);

        return true;
    } catch (const RedisExceiption& e) {
        std::cout << "RedisException:" << __LINE__ << ":" << __FUNCTION__ << ":" << e.what() << "\n";
        return false;
    }
}

int main()
{
    auto context = rediscpp::redisConnect(ip, port);
    if (!context) {
        std::cout << "error context error\n";
        return 0;
    }
    g_context = std::move(context);

    test();
    return 0;
}