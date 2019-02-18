#include "RedisCpp.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "TestTool.h"

std::string ip = "127.0.0.1";
int port = 6379;

using namespace rediscpp;

ContextGuard g_context;

bool test()
{
    try {
        Buffer key{ "a" };
        DEL(g_context, key);

        RedisZset redis{ g_context };
        TEST(redis.ZADD(key, 100, Buffer("b")) == 100);
        TEST(redis.ZADD(key, 100, Buffer("c")) == 100);
        TEST(redis.ZADD(key, 101, Buffer("c")) == 101);

        auto arr = redis.ZRANGE(key, 0, -1);
        pout(arr);

        auto arr_pair = redis.ZRANGE_WITHSCORES(key, 0, -1);
        for (const auto& val : arr_pair) {
            std::cout << val.first.asString() << " " << val.second.asString() << "\n";
        }
        return true;
    } catch (const RedisExceiption& e) {
        std::cout << "RedisException:" << __LINE__ << ":" << __FUNCTION__ << ":" << e.what() << "\n";
        return false;
    }
    return true;
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