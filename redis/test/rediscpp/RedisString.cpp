#include "RedisString.h"

#include "RedisException.h"
#include "Utile.h"

namespace rediscpp {

RedisString::RedisString(ContextGuard& context)
    : m_context(context)
{}

void RedisString::SET(Buffer key, Buffer value)
{
    ReplyGuard reply{ reinterpret_cast<redisReply*>(
        ::redisCommand(m_context.get(),"SET %b %b", key.getData(), key.getLen(), value.getData(), value.getLen())
        )
    };
    if (!reply)
        throw RedisReplyExceiption("SET reply null");
    if (reply->type == REDIS_REPLY_ERROR)
        throw RedisReplyExceiption(reply->str);
    if (reply->type != REDIS_REPLY_STATUS)
        throw RedisReplyExceiption("SET type REDIS_REPLY_STATUS");
    if (reply->str != std::string("OK"))
        throw RedisReplyExceiption("SET type REDIS_REPLY_STATUS OK");
}

Buffer RedisString::GET(Buffer key)
{
    ReplyGuard reply{ reinterpret_cast<redisReply*>(
        ::redisCommand(m_context.get(),"GET %b", key.getData(), key.getLen())
        )
    };
    if (!reply)
        throw RedisReplyExceiption("GET reply null");
    if (reply->type == REDIS_REPLY_NIL)
        return {};
    if (reply->type == REDIS_REPLY_ERROR)
        throw RedisReplyExceiption(reply->str);
    if (reply->type != REDIS_REPLY_STRING)
        throw RedisReplyExceiption("GET type REDIS_REPLY_STRING");
    return replyToRedisBuffer(reply.get());
}

long long RedisString::INCR(Buffer key)
{
    ReplyGuard reply{ reinterpret_cast<redisReply*>(
        ::redisCommand(m_context.get(),"INCR %b", key.getData(), key.getLen())
        )
    };
    if (!reply)
        throw RedisReplyExceiption("INCR reply null");
    if (reply->type == REDIS_REPLY_ERROR)
        throw RedisReplyExceiption(reply->str);
    if (reply->type != REDIS_REPLY_INTEGER)
        throw RedisReplyExceiption("INCR type REDIS_REPLY_INTEGER");
    return reply->integer;
}

long long RedisString::INCRBY(Buffer key, long long increment)
{
    ReplyGuard reply{ reinterpret_cast<redisReply*>(
        ::redisCommand(m_context.get(),"INCRBY %b %lld", key.getData(), key.getLen(), increment)
        )
    };
    if (!reply)
        throw RedisReplyExceiption("INCRBY reply null");
    if (reply->type == REDIS_REPLY_ERROR)
        throw RedisReplyExceiption(reply->str);
    if (reply->type != REDIS_REPLY_INTEGER)
        throw RedisReplyExceiption("INCRBY type REDIS_REPLY_INTEGER");
    return reply->integer;
}

}