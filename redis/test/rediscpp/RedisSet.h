#pragma once

#include "RedisType.h"

namespace rediscpp {

class Set
{
public:
    Set(ContextGuard& context);
    ~Set() = default;

    long long SADD(Buffer key, Buffer value);
    bool SISMEMBER(Buffer key, Buffer value);
private:
    ContextGuard& m_context;
};

}