#pragma once

#include <string>
#include <vector>
#include <map>
#include <list>
#include <hiredis.h>

#include "RedisType.h"

namespace rediscpp {

ContextGuard redisConnect(std::string ip, int port);
ContextGuard redisConnectWithTimeout(std::string ip, int port, int seconds, int microseconds);
int redisGetReply(redisContext* context, ReplyGuard* guard);

std::vector<std::string> splitString(std::string src, char c);

std::map<std::string, std::string> replyArrayToMap(const redisReply* reply, size_t count);
//std::vector<RedisKeyValue> replyArrayToKeyValue(const redisReply* reply, size_t count);
//std::vector<RedisKey> replyArrayToKey(const redisReply* reply, size_t count);
//std::vector<RedisValue> replyArrayToValue(const redisReply* reply, size_t count);

std::string replyToString(const void* p);
int32_t replyToStringInteger(const void* p);

//RedisKey replyToRedisKey(const void* p);
//RedisValue replyToRedisValue(const void* p);

Buffer replyToRedisBuffer(const void* p);

std::vector<std::pair<Buffer, Buffer>> replyArrayToPair(const redisReply* reply, size_t count);
std::vector<Buffer> replyArrayToBuffer(const redisReply* reply, size_t count);

/*
long long delKeys(RedisContextGuard& context, const std::list<std::string>& keys);
long long delKeys(RedisContextGuard& context, std::string key);
*/

long long DEL(ContextGuard& context, std::string key);
long long DEL(ContextGuard& context, std::vector<std::string> keys);
long long DEL(ContextGuard& context, Buffer key);
long long DEL(ContextGuard& context, std::vector<Buffer> keys);

std::string catFile(std::string path);
}
