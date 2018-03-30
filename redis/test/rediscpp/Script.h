#pragma once

#include "RedisType.h"

namespace rediscpp {

class Script
{
public:
    Script(ContextGuard& context);
    ~Script() = default;

    Buffer LOAD(Buffer cmd);

    BufferArray EVAL(Buffer cmd, std::vector<Buffer> keys = {}, std::vector<Buffer> values = {});
    BufferArray EVALSHA(Buffer cmd, std::vector<Buffer> keys = {}, std::vector<Buffer> values = {});
private:
    BufferArray evalInternal(std::string eval_cmd, Buffer cmd, std::vector<Buffer> keys, std::vector<Buffer> values);
    static BufferArray luaToRedis(const redisReply* reply);
private:
    ContextGuard& m_context;
};

}