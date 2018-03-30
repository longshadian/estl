#include "RedisType.h"
#include "RedisException.h"

#include <cstring>

namespace rediscpp {

Buffer::Buffer()
    : m_data()
{
}

Buffer::Buffer(const char* p)
    : Buffer(p, static_cast<int>(::strlen(p)))
{
}

Buffer::Buffer(const std::string& str)
{
    if (str.empty()) {
        return;
    }
    auto ptr = reinterpret_cast<const uint8_t*>(str.data());
    m_data.assign(ptr, ptr + str.size());
}

Buffer::Buffer(const char* p, int len)
{
    if (len <= 0)
        return;
    auto ptr = reinterpret_cast<const uint8_t*>(p);
    m_data.assign(ptr, ptr + len);
}

Buffer::Buffer(std::vector<uint8_t> data)
    : m_data(std::move(data))
{
}

Buffer::Buffer(const Buffer& rhs)
    : m_data(rhs.m_data)
{
}

Buffer::Buffer(int val)
    : Buffer(std::to_string(val))
{
}

Buffer::Buffer(unsigned int val)
    : Buffer(std::to_string(val))
{
}

Buffer::Buffer(long val)
    : Buffer(std::to_string(val))
{
}

Buffer::Buffer(unsigned long val)
    : Buffer(std::to_string(val))
{
}

Buffer::Buffer(long long val)
    : Buffer(std::to_string(val))
{
}

Buffer::Buffer(unsigned long long val)
    : Buffer(std::to_string(val))
{
}

Buffer::Buffer(float val)
    : Buffer(std::to_string(val))
{
}

Buffer::Buffer(double val)
    : Buffer(std::to_string(val))
{
}

Buffer::Buffer(long double val)
    : Buffer(std::to_string(val))
{
}

Buffer& Buffer::operator=(const Buffer& rhs)
{
    if (this != &rhs) {
        m_data = rhs.m_data;
    }
    return *this;
}

Buffer::Buffer(Buffer&& rhs)
    : m_data(std::move(rhs.m_data))
{
}

Buffer& Buffer::operator=(Buffer&& rhs)
{
    if (this != &rhs) {
        m_data = std::move(rhs.m_data);
    }
    return *this;
}

std::string Buffer::asString() const
{
    if (m_data.empty())
        return {};
    auto p = reinterpret_cast<const char*>(m_data.data());
    return std::string(p,  p + m_data.size());
}

int Buffer::asInt() const
{
    return asIntDetail<int>();
}

int8_t Buffer::asInt8() const
{
    return asIntDetail<int8_t>();
}

int16_t Buffer::asInt16() const
{
    return asIntDetail<int16_t>();
}

int32_t Buffer::asInt32() const
{
    return asIntDetail<int32_t>();
}

int64_t Buffer::asInt64() const
{
    return asIntDetail<int64_t>();
}

uint8_t Buffer::asUInt8() const
{
    return asIntDetail<uint8_t>();
}

uint16_t Buffer::asUInt16() const
{
    return asIntDetail<uint16_t>();
}

uint32_t Buffer::asUInt32() const
{
    return asIntDetail<uint32_t>();
}

uint64_t Buffer::asUInt64() const
{
    return asIntDetail<uint64_t>();
}

double Buffer::asDouble() const
{
    auto s = asString();
    return atof(s.c_str());
}

float Buffer::asFloat() const
{
    return static_cast<float>(asDouble());
}

const uint8_t* Buffer::getData() const
{
    return m_data.data();
}

size_t Buffer::getLen() const
{
    return m_data.size();
}

const std::vector<uint8_t>& Buffer::getDataVector() const
{
    return m_data;
}

bool Buffer::empty() const
{
    return m_data.empty();
}

void Buffer::clearBuffer()
{
    m_data.clear();
}

void Buffer::append(std::string s)
{
    append(s.data(), s.size());
}

void Buffer::append(const void* data, size_t len)
{
    auto p = reinterpret_cast<const uint8_t*>(data);
    for (size_t i = 0; i != len; ++i) {
        m_data.push_back(p[i]);
    }
}

//////////////////////////////////////////////////////////////////////////
BufferArray::BufferArray(TYPE t)
    : m_type(t)
    , m_buffer()
    , m_array()
{
}

BufferArray::BufferArray(const BufferArray& rhs)
    : m_type(rhs.m_type)
    , m_buffer(rhs.m_buffer)
    , m_array(rhs.m_array)
{
}

BufferArray& BufferArray::operator=(const BufferArray& rhs)
{
    if (this != &rhs) {
        m_type = rhs.m_type;
        m_buffer = rhs.m_buffer;
        m_array = rhs.m_array;
    }
    return *this;
}

BufferArray::BufferArray(BufferArray&& rhs)
    : m_type(rhs.m_type)
    , m_buffer(std::move(rhs.m_buffer))
    , m_array(std::move(rhs.m_array))
{
}

BufferArray& BufferArray::operator=(BufferArray&& rhs)
{
    if (this != &rhs) {
        m_type = rhs.m_type;
        m_buffer = std::move(rhs.m_buffer);
        m_array = std::move(rhs.m_array);
    }
    return *this;
}

BufferArray BufferArray::initBuffer()
{
    return BufferArray(TYPE::BUFFER);
}

BufferArray BufferArray::initArray()
{
    return BufferArray(TYPE::ARRAY);
}

bool BufferArray::isBuffer() const { return m_type == TYPE::BUFFER; }
bool BufferArray::isArray() const { return m_type == TYPE::ARRAY; }

void BufferArray::setBuffer(Buffer b)
{
    checkType(TYPE::BUFFER);
    m_buffer = std::move(b);
}

Buffer& BufferArray::getBuffer()&
{
    checkType(TYPE::BUFFER);
    return m_buffer;
}

const Buffer& BufferArray::getBuffer() const &
{
    checkType(TYPE::BUFFER);
    return m_buffer;
}

Buffer BufferArray::getBuffer() && 
{
    checkType(TYPE::BUFFER);
    return std::move(m_buffer);
}

void BufferArray::push_back(Buffer b)
{
    checkType(TYPE::ARRAY);
    auto a = BufferArray::initArray();
    a.setBuffer(std::move(b));
    m_array.push_back(std::move(a));
}

void BufferArray::push_back(BufferArray a)
{
    checkType(TYPE::ARRAY);
    m_array.push_back(std::move(a));
}

BufferArray BufferArray::pop_back()
{
    checkType(TYPE::ARRAY);
    auto a = std::move(m_array.back());
    m_array.pop_back();
    return a;
}

BufferArray& BufferArray::operator[](size_t idx) &
{
    checkType(TYPE::ARRAY);
    return m_array[idx];
}

const BufferArray& BufferArray::operator[](size_t idx) const &
{
    checkType(TYPE::ARRAY);
    return m_array[idx];
}

BufferArray BufferArray::operator[](size_t idx) &&
{
    checkType(TYPE::ARRAY);
    return std::move(m_array[idx]);
}

size_t BufferArray::size() const
{
    checkType(TYPE::ARRAY);
    return m_array.size();
}

bool BufferArray::empty() const
{
    checkType(TYPE::ARRAY);
    return m_array.empty();
}

BufferArray::iterator BufferArray::begin()
{
    checkType(TYPE::ARRAY);
    return m_array.begin();
}

BufferArray::iterator BufferArray::end()
{
    checkType(TYPE::ARRAY);
    return m_array.end();
}

BufferArray::const_iterator BufferArray::begin() const
{
    return cbegin();
}

BufferArray::const_iterator BufferArray::end() const
{
    return cend();
}

BufferArray::const_iterator BufferArray::cbegin() const
{
    checkType(TYPE::ARRAY);
    return m_array.cbegin();
}

BufferArray::const_iterator BufferArray::cend() const
{
    checkType(TYPE::ARRAY);
    return m_array.cend();
}

void BufferArray::checkType(TYPE t) const
{
    if (m_type != t) {
        if (t == TYPE::BUFFER)
            throw RedisExceiption("BufferArray is not BUFFER!");
        else if (t == TYPE::ARRAY)
            throw RedisExceiption("BufferArray is not ARRAY!");
    }
}

}
