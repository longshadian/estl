#include "RedisHash.h"

#include "RedisException.h"
#include "Utile.h"

namespace rediscpp {

RedisHash::RedisHash(ContextGuard& context)
    : m_context(context)
{

}

std::vector<std::pair<Buffer, Buffer>> RedisHash::HGETALL(Buffer key)
{
    ReplyGuard reply{ reinterpret_cast<redisReply*>(
        ::redisCommand(m_context.get(),"HGETALL %b", key.getData(), key.getLen())
        )
    };
    if (!reply)
        throw RedisReplyExceiption("HGETALL reply null");
    if (reply->type == REDIS_REPLY_ERROR)
        throw RedisReplyExceiption(reply->str);
    if (reply->type != REDIS_REPLY_ARRAY)
        throw RedisReplyExceiption("HGETALL type REDIS_REPLY_ARRAY");
    return replyArrayToPair(reply.get(), reply->elements);
}

//由给定数量增加的哈希字段的整数值
long long RedisHash::HINCRBY(Buffer key, Buffer mkey, long long increment)
{
    ReplyGuard reply{ reinterpret_cast<redisReply*>(
        ::redisCommand(m_context.get(),"HINCRBY %b %b %lld", key.getData(), key.getLen(),
            mkey.getData(), mkey.getLen(), increment)
        )
    };
    if (!reply)
        throw RedisReplyExceiption("HINCRBY reply null");
    if (reply->type == REDIS_REPLY_ERROR)
        throw RedisReplyExceiption(reply->str);
    if (reply->type != REDIS_REPLY_INTEGER)
        throw RedisReplyExceiption("HINCRBY type REDIS_REPLY_INTEGER");
    return reply->integer;
}

//设置哈希字段的字符串值, return 0 1
long long RedisHash::HSET(Buffer key, Buffer mkey, Buffer value)
{
    ReplyGuard reply{ reinterpret_cast<redisReply*>(
        ::redisCommand(m_context.get(),"HSET %b %b %b", key.getData(), key.getLen(),
            mkey.getData(), mkey.getLen(), value.getData(), value.getLen())
        )
    };
    if (!reply)
        throw RedisReplyExceiption("HSET reply null");
    if (reply->type == REDIS_REPLY_ERROR)
        throw RedisReplyExceiption(reply->str);
    if (reply->type != REDIS_REPLY_INTEGER)
        throw RedisReplyExceiption("HSET type REDIS_REPLY_INTEGER");
    return reply->integer;
}

Buffer RedisHash::HGET(Buffer key, Buffer mkey)
{
    ReplyGuard reply{ reinterpret_cast<redisReply*>(
        ::redisCommand(m_context.get(), "HGET %b %b",
            key.getData(), key.getLen(),
            mkey.getData(), mkey.getLen())
        )
    };
    if (!reply)
        throw RedisReplyExceiption("HGET reply null");
    if (reply->type == REDIS_REPLY_NIL)
        return{};
    if (reply->type == REDIS_REPLY_ERROR)
        throw RedisReplyExceiption(reply->str);
    if (reply->type != REDIS_REPLY_STRING)
        throw RedisReplyExceiption("HGET type REDIS_REPLY_STRING");
    return replyToRedisBuffer(reply.get());
}

long long RedisHash::HDEL(Buffer key, Buffer mkey)
{
    ReplyGuard reply{ reinterpret_cast<redisReply*>(
        ::redisCommand(m_context.get(),"HDEL %b %b", key.getData(), key.getLen(),
            mkey.getData(), mkey.getLen())
        )
    };
    if (!reply)
        throw RedisReplyExceiption("HDEL reply null");
    if (reply->type == REDIS_REPLY_ERROR)
        throw RedisReplyExceiption(reply->str);
    if (reply->type != REDIS_REPLY_INTEGER)
        throw RedisReplyExceiption("HDEL type REDIS_REPLY_INTEGER");
    return reply->integer;
}

}