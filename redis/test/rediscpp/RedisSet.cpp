#include "RedisSet.h"

#include "RedisException.h"
#include "Utile.h"

namespace rediscpp {

Set::Set(ContextGuard& context)
    : m_context(context)
{}

long long Set::SADD(Buffer key, Buffer value)
{
    ReplyGuard reply{ reinterpret_cast<redisReply*>(
        ::redisCommand(m_context.get(),"SADD %b %b", key.getData(), key.getLen(),
            value.getData(), value.getLen())
        )
    };
    if (!reply)
        throw RedisReplyExceiption("SADD reply null");
    if (reply->type == REDIS_REPLY_ERROR) 
        throw RedisReplyExceiption(reply->str);
    if (reply->type != REDIS_REPLY_INTEGER)
        throw RedisReplyExceiption("SADD REDIS_REPLY_INTEGER");
    return reply->integer;
}

bool Set::SISMEMBER(Buffer key, Buffer value)
{
    ReplyGuard reply{ reinterpret_cast<redisReply*>(
        ::redisCommand(m_context.get(),"SISMEMBER %b %b", key.getData(), key.getLen(),
            value.getData(), value.getLen())
        )
    };
    if (!reply)
        throw RedisReplyExceiption("SISMEMBER reply null");
    if (reply->type == REDIS_REPLY_ERROR) 
        throw RedisReplyExceiption(reply->str);
    if (reply->type != REDIS_REPLY_INTEGER)
        throw RedisReplyExceiption("SISMEMBER REDIS_REPLY_INTEGER");
    return reply->integer == 1;
}

}