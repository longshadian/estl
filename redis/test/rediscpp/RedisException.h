#pragma once

#include <exception>
#include <string>

namespace rediscpp {

class RedisExceiption : public std::exception
{
public:
    RedisExceiption(std::string s);
    ~RedisExceiption() throw() {}

    virtual const char* what() const noexcept override { return m_msg.c_str(); }
private:
    std::string m_msg;
};

class RedisReplyExceiption : public RedisExceiption
{
public:
    RedisReplyExceiption(const char* str);
    RedisReplyExceiption(std::string str);
    ~RedisReplyExceiption() throw() {}
};

class RedisReplyTypeExceiption : public RedisExceiption
{
public:
    RedisReplyTypeExceiption(const char* str);
    RedisReplyTypeExceiption(std::string str);
    ~RedisReplyTypeExceiption() throw() {}
};


}