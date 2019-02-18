#pragma once

#include <hiredis.h>

#include <string>
#include <vector>
#include <memory>

namespace rediscpp {

class Buffer
{
public:
    Buffer();
    ~Buffer() = default;

    explicit Buffer(const char* p);
    explicit Buffer(const std::string& str);
    explicit Buffer(const char* p, int len);
    explicit Buffer(std::vector<uint8_t> data);
    explicit Buffer(int val);
    explicit Buffer(unsigned int val);
    explicit Buffer(long val);
    explicit Buffer(unsigned long val);
    explicit Buffer(long long val);
    explicit Buffer(unsigned long long val);
    explicit Buffer(float val);
    explicit Buffer(double val);
    explicit Buffer(long double val);

    Buffer(const Buffer& rhs);
    Buffer& operator=(const Buffer& rhs);

    Buffer(Buffer&& rhs);
    Buffer& operator=(Buffer&& rhs);

    std::string asString() const;

    int asInt() const;
    int8_t asInt8() const;
    int16_t asInt16() const;
    int32_t asInt32() const;
    int64_t asInt64() const;

    uint8_t asUInt8() const;
    uint16_t asUInt16() const;
    uint32_t asUInt32() const;
    uint64_t asUInt64() const;

    double asDouble() const;
    float asFloat() const;

    const uint8_t* getData() const;
    size_t getLen() const;
    const std::vector<uint8_t>& getDataVector() const;
    bool empty() const;
    void clearBuffer();

    void append(std::string s);
    void append(const void* data, size_t len);
private:
    template <typename T>
    T asIntDetail() const
    {
        static_assert(std::is_integral<T>::value, "T must be Integeral!");
        auto s = asString();
        if (s.empty())
            return T();
        return static_cast<T>(std::atoll(s.c_str()));
    }
private:
    std::vector<uint8_t>    m_data;
};

class BufferArray
{
public:
    enum class TYPE 
    {
        BUFFER = 0,
        ARRAY = 1,
    };
private:
    BufferArray(TYPE t);
public:
    using array_type    = std::vector<BufferArray>;
    using iterator      = array_type::iterator;
    using const_iterator = array_type::const_iterator;      

    ~BufferArray() = default;
    BufferArray(const BufferArray& rhs); 
    BufferArray& operator=(const BufferArray& rhs); 
    BufferArray(BufferArray&& rhs); 
    BufferArray& operator=(BufferArray&& rhs); 

    static BufferArray initBuffer();
    static BufferArray initArray();

    bool isBuffer() const;
    bool isArray() const;

    //BUFFER ²Ù×÷
    void setBuffer(Buffer b);
    Buffer& getBuffer()&;
    const Buffer& getBuffer() const &;
    Buffer getBuffer() &&;

public:
    //ARRAY ²Ù×÷

    void                push_back(Buffer b);
    void                push_back(BufferArray a);
    BufferArray         pop_back();

    BufferArray&        operator[](size_t idx) &;
    const BufferArray&  operator[](size_t idx) const &;
    BufferArray         operator[](size_t idx) &&;

    size_t              size() const;
    bool                empty() const;
    iterator            begin();
    iterator            end();
    const_iterator      begin() const;
    const_iterator      end() const;
    const_iterator      cbegin() const;
    const_iterator      cend() const;
private:
    void checkType(TYPE t) const;
private:
    TYPE                     m_type;
    Buffer                   m_buffer;
    std::vector<BufferArray> m_array;
};


struct FreeReplyObject
{
    void operator()(redisReply* x)
    {
        if (x)
            freeReplyObject(x);
    }
};

struct RedisFree
{
    void operator()(redisContext* x)
    {
        if (x)
            redisFree(x);
    }
};

typedef std::unique_ptr<redisContext, RedisFree> ContextGuard;

typedef std::unique_ptr<redisReply, FreeReplyObject> ReplyGuard;

}