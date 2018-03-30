#include "Script.h"
#include <iostream>
#include <cstring>

#include "RedisException.h"
#include "Utile.h"

namespace rediscpp {

Script::Script(ContextGuard& context)
    : m_context(context)
{}

Buffer Script::LOAD(Buffer cmd)
{
    ReplyGuard reply{ reinterpret_cast<redisReply*>(
        ::redisCommand(m_context.get(),"SCRIPT LOAD %b", cmd.getData(), cmd.getLen())
        )
    };
    if (!reply)
        throw RedisReplyExceiption("SCRIPT LOAD reply null");
    if (reply->type == REDIS_REPLY_ERROR) 
        throw RedisReplyExceiption(reply->str);
    if (reply->type != REDIS_REPLY_STRING)
        throw RedisReplyExceiption("SCRIPT LOAD REDIS_REPLY_STRING");

    return replyToRedisBuffer(reply.get());
}

BufferArray Script::EVAL(Buffer cmd, std::vector<Buffer> keys, std::vector<Buffer> values)
{
    std::string evalsha = "EVAL";
    return evalInternal(std::move(evalsha), std::move(cmd), std::move(keys), std::move(values));
}

BufferArray Script::EVALSHA(Buffer cmd, std::vector<Buffer> keys,
    std::vector<Buffer> values)
{
    std::string evalsha = "EVALSHA";
    return evalInternal(std::move(evalsha), std::move(cmd), std::move(keys), std::move(values));
}

BufferArray Script::evalInternal(std::string eval_cmd, Buffer cmd,
    std::vector<Buffer> keys, std::vector<Buffer> values)
{
    std::vector<std::vector<uint8_t>> final_buffer;
    std::vector<uint8_t> temp_buffer;

    temp_buffer.resize(eval_cmd.size());
    std::memcpy(temp_buffer.data(), eval_cmd.data(), eval_cmd.size());
    final_buffer.emplace_back(std::move(temp_buffer));

    temp_buffer.resize(cmd.getLen());
    std::memcpy(temp_buffer.data(), cmd.getData(), cmd.getLen());
    final_buffer.emplace_back(std::move(temp_buffer));

    auto keys_count = std::to_string(keys.size());
    temp_buffer.resize(keys_count.size());
    std::memcpy(temp_buffer.data(), keys_count.data(), keys_count.size());
    final_buffer.emplace_back(std::move(temp_buffer));

    for (size_t i = 0; i != keys.size(); ++i) {
        const auto& key = keys[i];
        temp_buffer.resize(key.getLen());
        std::memcpy(temp_buffer.data(), key.getData(), key.getLen());
        final_buffer.emplace_back(std::move(temp_buffer));
    }

    for (size_t i = 0; i != values.size(); ++i) {
        const auto& val = values[i];
        temp_buffer.resize(val.getLen());
        std::memcpy(temp_buffer.data(), val.getData(), val.getLen());
        final_buffer.emplace_back(std::move(temp_buffer));
    }

    std::vector<const char*> argv;
    std::vector<size_t> arglen;
    for (size_t i = 0; i != final_buffer.size(); ++i) {
        argv.push_back(reinterpret_cast<const char*>(final_buffer[i].data()));
        arglen.push_back(final_buffer[i].size());
    }

    ReplyGuard reply{ 
        reinterpret_cast<redisReply*>(
            ::redisCommandArgv(m_context.get(), static_cast<int>(argv.size()), argv.data(), arglen.data()))
    };
    if (!reply)
        throw RedisReplyExceiption("EVALSHA reply null");
    if (reply->type == REDIS_REPLY_ERROR)
        throw RedisReplyExceiption(reply->str);
    return luaToRedis(reply.get());
}

BufferArray Script::luaToRedis(const redisReply* reply)
{
    /*
    Lua to Redis conversion table.
    Lua number->Redis integer reply(the number is converted into an integer)
    Lua string->Redis bulk reply
    Lua table(array)->Redis multi bulk reply(truncated to the first nil inside the Lua array if any)
    Lua table with a single ok field->Redis status reply
    Lua table with a single err field->Redis error reply
    Lua boolean false->Redis Nil bulk reply.
    There is an additional Lua - to - Redis conversion rule that has no corresponding Redis to Lua conversion rule :
    Lua boolean true->Redis integer reply with value of 1.
    */

    switch (reply->type) {
    case REDIS_REPLY_STRING: {
        auto ret = BufferArray::initBuffer();
        ret.setBuffer(replyToRedisBuffer(reply));
        return ret;
    }
    case REDIS_REPLY_ARRAY: {
        auto ret = BufferArray::initArray();
        for (size_t i = 0; i != reply->elements; ++i) {
            ret.push_back(luaToRedis(reply->element[i]));
        }
        return ret;
    }
    case REDIS_REPLY_INTEGER: {
        auto ret = BufferArray::initBuffer();
        ret.setBuffer(Buffer(reply->integer));
        return ret;
    }
    case REDIS_REPLY_NIL: {
        return BufferArray::initBuffer();
    }
    case REDIS_REPLY_STATUS: {
        auto ret = BufferArray::initBuffer();
        ret.setBuffer(replyToRedisBuffer(reply));
        return ret;
    }
    case REDIS_REPLY_ERROR: {
        throw RedisReplyExceiption(reply->str);
    }
    default:
        break;
    }
    return BufferArray::initBuffer();
}


}
