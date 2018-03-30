#pragma once

#include "RedisType.h"

namespace rediscpp {

class RedisHash
{
public:
    RedisHash(ContextGuard& context);
    ~RedisHash() = default;

    //删除一个或多个哈希字段
    //int HDEL(RedisKey key, RedisKey field);

    //让所有的字段和值在指定的键存储在一个哈希
    std::vector<std::pair<Buffer, Buffer>> HGETALL(Buffer key);

    //由给定数量增加的哈希字段的整数值
    long long HINCRBY(Buffer key, Buffer mkey, long long increment);

    //设置哈希字段的字符串值, return 0 1
    long long HSET(Buffer key, Buffer mkey, Buffer value);

    Buffer HGET(Buffer key, Buffer mkey);

    long long HDEL(Buffer key, Buffer mkey);
private:
    ContextGuard& m_context;
};

}