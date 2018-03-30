#include "RedisException.h"

namespace rediscpp {

RedisExceiption::RedisExceiption(std::string s)
    : m_msg(std::move(s))
{
}

RedisReplyExceiption::RedisReplyExceiption(const char* str)
    : RedisExceiption(str)
{
}

RedisReplyExceiption::RedisReplyExceiption(std::string str)
    : RedisExceiption(std::move(str))
{
}

RedisReplyTypeExceiption::RedisReplyTypeExceiption(const char* str)
    : RedisExceiption(str)
{
}

RedisReplyTypeExceiption::RedisReplyTypeExceiption(std::string str)
    : RedisExceiption(std::move(str))
{
}

}
