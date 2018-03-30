#include "Utile.h"

#include <sstream>
#include <fstream>
#include "RedisException.h"

namespace rediscpp {

ContextGuard redisConnect(std::string ip, int port)
{
    return ContextGuard(::redisConnect(ip.c_str(), port));
}

ContextGuard redisConnectWithTimeout(std::string ip, int port, int seconds, int microseconds)
{
    struct timeval tv;
    tv.tv_sec = seconds;
    tv.tv_usec = microseconds;
    return ContextGuard(::redisConnectWithTimeout(ip.c_str(), port, tv));
}

int redisGetReply(redisContext* context, ReplyGuard* guard)
{
    redisReply* reply = nullptr;
    int ret = ::redisGetReply(context, (void**)&reply);
    if (ret != REDIS_OK)
        return ret;
    *guard = ReplyGuard(reply);
    return ret;
}


std::vector<std::string> splitString(std::string src, char c)
{
    std::vector<std::string> out;
    if (src.empty())
        return out;
    std::istringstream istm(src);
    std::string temp;
    while (std::getline(istm, temp, c)) {
        out.push_back(temp);
    }
    return out;
}

std::map<std::string, std::string> replyArrayToMap(const redisReply* reply, size_t count)
{
    if (count % 2 != 0) {
        throw RedisReplyTypeExceiption("Array count");
    }

    std::map<std::string, std::string> ret;
    for (size_t i = 0; i != count; ++i) {
        std::string key = replyToString(reply->element[i]);
        ++i;
        if (i >= count) {
            throw RedisReplyTypeExceiption("Array count");
        }
        std::string val = replyToString(reply->element[i]);
        ret.insert({key, val});
    }
    return ret;
}

/*
std::vector<RedisKeyValue> replyArrayToKeyValue(const redisReply* reply, size_t count)
{
    if (count % 2 != 0) {
        throw RedisReplyTypeExceiption("replyArrayToKeyValue count");
    }

    std::vector<RedisKeyValue> ret;
    for (size_t i = 0; i != count; ++i) {
        RedisKeyValue temp;
        temp.m_key = replyToRedisKey(reply->element[i]);
        ++i;
        if (i >= count) {
            throw RedisReplyTypeExceiption("Array count");
        }
        temp.m_value = replyToRedisValue(reply->element[i]);
        ret.emplace_back(std::move(temp));
    }
    return ret;
}

std::vector<RedisKey> replyArrayToKey(const redisReply* reply, size_t count)
{
    std::vector<RedisKey> ret;
    for (size_t i = 0; i != count; ++i) {
        ret.emplace_back(replyToRedisKey(reply->element[i]));
    }
    return ret;
}

std::vector<RedisValue> replyArrayToValue(const redisReply* reply, size_t count)
{
    std::vector<RedisValue> ret;
    for (size_t i = 0; i != count; ++i) {
        ret.emplace_back(replyToRedisValue(reply->element[i]));
    }
    return ret;
}
*/

std::string replyToString(const void* p)
{
    auto reply = reinterpret_cast<const redisReply*>(p);
    std::string s(reply->str, reply->len);
    return s;
}

int32_t replyToStringInteger(const void* p)
{
    auto s = replyToString(p);
    return atoi(s.c_str());
}

/*
RedisKey replyToRedisKey(const void* p)
{
    auto reply = reinterpret_cast<const redisReply*>(p);
    return RedisKey(reply->str, reply->len);
}

RedisValue replyToRedisValue(const void* p)
{
    auto reply = reinterpret_cast<const redisReply*>(p);
    return RedisValue(reply->str, reply->len);
}
*/

Buffer replyToRedisBuffer(const void* p)
{
    auto reply = reinterpret_cast<const redisReply*>(p);
    return Buffer(reply->str, reply->len);
}

std::vector<std::pair<Buffer, Buffer>> replyArrayToPair(const redisReply* reply, size_t count)
{
    if (count % 2 != 0) {
        throw RedisReplyTypeExceiption("replyArrayToPair count error");
    }

    std::vector<std::pair<Buffer, Buffer>> ret;
    for (size_t i = 0; i != count; ++i) {
        Buffer key = replyToRedisBuffer(reply->element[i]);
        ++i;
        if (i >= count) {
            throw RedisReplyTypeExceiption("Array count error");
        }
        Buffer value = replyToRedisBuffer(reply->element[i]);
        ret.emplace_back(std::make_pair(key, value));
    }
    return ret;
}

std::vector<Buffer> replyArrayToBuffer(const redisReply* reply, size_t count)
{
    std::vector<Buffer> ret;
    for (size_t i = 0; i != count; ++i) {
        ret.emplace_back(replyToRedisBuffer(reply->element[i]));
    }
    return ret;
}

/*
long long delKeys(RedisContextGuard& context, std::string key)
{
    return delKeys(context, std::list<std::string>{key});
}

long long delKeys(RedisContextGuard& context, const std::list<std::string>& keys)
{
    if (keys.empty()) {
        return 0;
    }

    std::ostringstream ostm;
    ostm << "DEL ";
    for (auto it : keys) {
        ostm << ' ' << it;
    }

    std::string cmd = ostm.str();
    RedisReplyGuard reply(reinterpret_cast<redisReply*>(::redisCommand(context.get(), cmd.c_str())));
    if (!reply || reply->type != REDIS_REPLY_INTEGER) {
        throw RedisReplyExceiption("DEL error");
    }
    return reply->integer;
}
*/

long long DEL(ContextGuard& context, std::string key)
{
    std::vector<std::string> temp;
    temp.emplace_back(std::move(key));
    return DEL(context, temp);
}

long long DEL(ContextGuard& context, std::vector<std::string> keys)
{
    std::vector<Buffer> temp;
    for (const auto& it : keys) {
        temp.emplace_back(Buffer(it));
    }
    return DEL(context, temp);
}

long long DEL(ContextGuard& context, Buffer key)
{
    std::vector<Buffer> temp;
    temp.emplace_back(std::move(key));
    return DEL(context, std::move(temp));
}

long long DEL(ContextGuard& context, std::vector<Buffer> keys)
{
    if (keys.empty()) {
        return 0;
    }

    std::ostringstream ostm;
    ostm << "DEL ";
    for (const auto& it : keys) {
        ostm << ' ' << it.asString();
    }

    std::string cmd = ostm.str();
    ReplyGuard reply{ reinterpret_cast<redisReply*>(::redisCommand(context.get(), cmd.c_str()))};
    if (!reply)
        throw RedisReplyExceiption("DEL reply null");
    if (reply->type == REDIS_REPLY_ERROR)
        throw RedisReplyExceiption(reply->str);
    if (reply->type != REDIS_REPLY_INTEGER)
        throw RedisReplyExceiption("DEL type REDIS_REPLY_INTEGER");
    return reply->integer;
}

std::string catFile(std::string path)
{
    std::string context;
    std::ifstream ifsm(path);
    if (!ifsm)
        return {};
    auto a = ifsm.get();
    while (a != EOF) {
        context.push_back(static_cast<char>(a));
        a = ifsm.get();
    }
    return context;
}

}
