#pragma once

#include "RedisType.h"

namespace rediscpp {

class RedisZset
{
public:
    RedisZset(ContextGuard& context);
    ~RedisZset() = default;

    long long ZADD(Buffer key, long long score, Buffer value);

    //有序集合范围内的成员和分数
    std::vector<Buffer> ZRANGE(Buffer key, int start, int end);
    std::vector<std::pair<Buffer, Buffer>> ZRANGE_WITHSCORES(Buffer key, int start, int end);

    //有序集合范围内的成员和分数,降序
    //first:    value
    //second:   score
    std::vector<Buffer> ZREVRANGE(Buffer key, int start, int end);
    std::vector<std::pair<Buffer, Buffer>> ZREVRANGE_WITHSCORES(Buffer key, int start, int end);

    //在有序集合增加成员的分数
    long long ZINCRBY(Buffer key, long long increment, Buffer value);
private:
    ContextGuard& m_context;
};

}