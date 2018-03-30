#pragma once

#include "RedisType.h"

namespace rediscpp {

class RedisZset
{
public:
    RedisZset(ContextGuard& context);
    ~RedisZset() = default;

    long long ZADD(Buffer key, long long score, Buffer value);

    //���򼯺Ϸ�Χ�ڵĳ�Ա�ͷ���
    std::vector<Buffer> ZRANGE(Buffer key, int start, int end);
    std::vector<std::pair<Buffer, Buffer>> ZRANGE_WITHSCORES(Buffer key, int start, int end);

    //���򼯺Ϸ�Χ�ڵĳ�Ա�ͷ���,����
    //first:    value
    //second:   score
    std::vector<Buffer> ZREVRANGE(Buffer key, int start, int end);
    std::vector<std::pair<Buffer, Buffer>> ZREVRANGE_WITHSCORES(Buffer key, int start, int end);

    //�����򼯺����ӳ�Ա�ķ���
    long long ZINCRBY(Buffer key, long long increment, Buffer value);
private:
    ContextGuard& m_context;
};

}