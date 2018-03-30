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

bool test_Set()
{
    try {
        Buffer key{"test_set"};
        DEL(g_context, key);
        
        Set redis{g_context};
        TEST(redis.SADD(key, Buffer("a")) == 1);
        TEST(redis.SADD(key, Buffer("b")) == 1);
        TEST(redis.SADD(key, Buffer("c")) == 1);
        TEST(redis.SADD(key, Buffer("c")) == 0);

        TEST(redis.SISMEMBER(key, Buffer("a")));
        TEST(!redis.SISMEMBER(key, Buffer("aa")));
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

    test_Set();
    return 0;
}