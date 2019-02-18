#pragma once

#include "RedisType.h"

namespace rediscpp {

class RedisString
{
public:
    RedisString(ContextGuard& context);
    ~RedisString() = default;

    void SET(Buffer key, Buffer value);
    Buffer GET(Buffer key);

    long long INCR(Buffer key);
    long long INCRBY(Buffer key, long long increment);
private:
    ContextGuard& m_context;
};

}