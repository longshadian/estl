#include <hiredis/hiredis.h>

#include <string>
#include <iostream>


int main()
{
    redisContext* redis = ::redisConnect("127.0.0.1", 6379);
    if (!redis) {
        printf("ERROR: conn");
        return 0;
    }

    redisReply* reply = (redisReply*)::redisCommand(redis, "SET %f %s", 12.3, "hhh345");
    if (!reply) {
        printf("ERROR: reply");
        return 0;
    }
    ::freeReplyObject(reply);
    reply = nullptr;

    reply = (redisReply*)::redisCommand(redis, "GET %f", 12.3);
    if (!reply) {
        printf("ERROR: reply");
        return 0;
    }

    printf("type:%d\n", reply->type);
    printf("integer:%lld\n", reply->integer);
    printf("len:%d\n", reply->len);
    printf("str:%s\n", reply->str);
    printf("elements:%d\n", (int)reply->elements);

    ::freeReplyObject(reply);
    reply = nullptr;
    return 0;
}
