#include "RedisList.h"

#include "RedisException.h"
#include "Utile.h"

namespace rediscpp {
RedisList::RedisList(ContextGuard& context)
    : m_context(context)
{}

long long RedisList::LLEN(Buffer key)
{
    ReplyGuard reply{reinterpret_cast<redisReply*>(
        ::redisCommand(m_context.get(),"LLEN %b", key.getData(), key.getLen()))};
    if (!reply)
        throw RedisReplyExceiption("LLEN reply null");
    if (reply->type == REDIS_REPLY_ERROR)
        throw RedisReplyExceiption(reply->str);
    if (reply->type != REDIS_REPLY_INTEGER)
        throw RedisReplyExceiption("LLEN REDIS_REPLY_INTEGER");
    return reply->integer;
}

Buffer RedisList::LPOP(Buffer key)
{
    ReplyGuard reply{reinterpret_cast<redisReply*>(
        ::redisCommand(m_context.get(),"LPOP %b", key.getData(), key.getLen()))};
    if (!reply)
        throw RedisReplyExceiption("LPOP null");
    if (reply->type == REDIS_REPLY_NIL)
        return {};
    if (reply->type == REDIS_REPLY_ERROR)
        throw RedisReplyExceiption(reply->str);
    if (reply->type != REDIS_REPLY_STRING)
        throw RedisReplyExceiption("LPOP REDIS_REPLY_STRING");
    return replyToRedisBuffer(reply.get());
}

Buffer RedisList::RPOP(Buffer key)
{
    ReplyGuard reply{reinterpret_cast<redisReply*>(
        ::redisCommand(m_context.get(),"RPOP %b", key.getData(), key.getLen()))};
    if (!reply)
        throw RedisReplyExceiption("RPOP null");
    if (reply->type == REDIS_REPLY_NIL)
        return Buffer();
    if (reply->type == REDIS_REPLY_ERROR)
        throw RedisReplyExceiption(reply->str);
    if (reply->type != REDIS_REPLY_STRING)
        throw RedisReplyExceiption("RPOP REDIS_REPLY_STRING");
    return replyToRedisBuffer(reply.get());
}

std::vector<Buffer> RedisList::LRANGE(Buffer key, int start, int stop)
{
    ReplyGuard reply{reinterpret_cast<redisReply*>(
        ::redisCommand(m_context.get(), "LRANGE %b %d %d", key.getData(), key.getLen(), start, stop))};
    if (!reply)
        throw RedisReplyExceiption("LRANGE null");
    if (reply->type == REDIS_REPLY_ERROR)
        throw RedisReplyExceiption(reply->str);
    if (reply->type != REDIS_REPLY_ARRAY)
        throw RedisReplyExceiption("LRANGE REDIS_REPLY_ARRAY");
    return replyArrayToBuffer(reply.get(), reply->elements);
}

long long RedisList::LPUSH(Buffer key, Buffer val)
{
    ReplyGuard reply{reinterpret_cast<redisReply*>(
        ::redisCommand(m_context.get(), "LPUSH %b %b", key.getData(), key.getLen(), val.getData(), val.getLen()))};
    if (!reply)
        throw RedisReplyExceiption("LPUSH reply null");
    if (reply->type == REDIS_REPLY_ERROR)
        throw RedisReplyExceiption(reply->str);
    if (reply->type != REDIS_REPLY_INTEGER)
        throw RedisReplyExceiption("LPUSH REDIS_REPLY_INTEGER");
    return reply->integer;
}

long long RedisList::RPUSH(Buffer key, Buffer val)
{
    ReplyGuard reply{reinterpret_cast<redisReply*>(
        ::redisCommand(m_context.get(), "RPUSH %b %b", key.getData(), key.getLen(),
            val.getData(), val.getLen()))};
    if (!reply)
        throw RedisReplyExceiption("RPUSH reply null");
    if (reply->type == REDIS_REPLY_ERROR)
        throw RedisReplyExceiption(reply->str);
    if (reply->type != REDIS_REPLY_INTEGER)
        throw RedisReplyExceiption("RPUSH REDIS_REPLY_INTEGER");
    return reply->integer;
}

}