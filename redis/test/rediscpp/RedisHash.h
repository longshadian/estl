#pragma once

#include "RedisType.h"

namespace rediscpp {

class RedisHash
{
public:
    RedisHash(ContextGuard& context);
    ~RedisHash() = default;

    //ɾ��һ��������ϣ�ֶ�
    //int HDEL(RedisKey key, RedisKey field);

    //�����е��ֶκ�ֵ��ָ���ļ��洢��һ����ϣ
    std::vector<std::pair<Buffer, Buffer>> HGETALL(Buffer key);

    //�ɸ����������ӵĹ�ϣ�ֶε�����ֵ
    long long HINCRBY(Buffer key, Buffer mkey, long long increment);

    //���ù�ϣ�ֶε��ַ���ֵ, return 0 1
    long long HSET(Buffer key, Buffer mkey, Buffer value);

    Buffer HGET(Buffer key, Buffer mkey);

    long long HDEL(Buffer key, Buffer mkey);
private:
    ContextGuard& m_context;
};

}