#pragma once

#include <exception>
#include <list>
#include <map>
#include <string>
#include <vector>
#include <cstring>
#include <cstdint>
#include <time.h>
#include <cmath>
#include <type_traits>
#include <cassert>

namespace zylib {

class ByteBufferException : public std::runtime_error
{
public:
    ~ByteBufferException() throw() { }

    char const* what() const throw() override { return m_msg.c_str(); }

protected:
    std::string & message() throw() { return m_msg; }

private:
    std::string m_msg;
};

class ByteBufferPositionException : public ByteBufferException
{
public:
    ByteBufferPositionException(bool add, size_t pos, size_t size, size_t valueSize);

    ~ByteBufferPositionException() throw() { }
};

class ByteBufferSourceException : public ByteBufferException
{
public:
    ByteBufferSourceException(size_t pos, size_t size, size_t valueSize);

    ~ByteBufferSourceException() throw() { }
};

class ByteBuffer
{
public:
    static const size_t DEFAULT_SIZE = 1024 * 4;

    ByteBuffer()
        :ByteBuffer(DEFAULT_SIZE)
    {
    }

    ByteBuffer(size_t res) 
        : m_rpos(0), m_wpos(0), m_storage()
    {
        m_storage.reserve(res);
    }

    ByteBuffer(std::vector<uint8_t> buf) 
        : m_rpos(0), m_wpos(0), m_storage(std::move(buf)) 
    {
    }

    ByteBuffer(ByteBuffer&& buf) 
        : m_rpos(buf.m_rpos), m_wpos(buf.m_wpos), m_storage(std::move(buf.m_storage)) 
    {
    }

    ByteBuffer(const ByteBuffer& right) 
        : m_rpos(right.m_rpos), m_wpos(right.m_wpos), m_storage(right.m_storage) 
    {
    }

    ByteBuffer(const void* src, size_t len) 
        : m_rpos(0), m_wpos(0), m_storage(static_cast<const uint8_t*>(src), static_cast<const uint8_t*>(src) + len) 
    {
    }

    ByteBuffer& operator=(const ByteBuffer& right)
    {
        if (this != &right) {
            m_rpos = right.m_rpos;
            m_wpos = right.m_wpos;
            m_storage = right.m_storage;
        }
        return *this;
    }

    ByteBuffer& operator=(ByteBuffer&& right) 
    {
        if (this != &right) {
            m_rpos = right.m_rpos;
            m_wpos = right.m_wpos;
            m_storage = std::move(right.m_storage);
        }
        return *this;
    }

    ~ByteBuffer() 
    {
    }

    void clear()
    {
        m_rpos = 0;
        m_wpos = 0;
        m_storage.clear();
    }

    template <typename T> 
    void append(T value)
    {
        static_assert(std::is_fundamental<T>::value, "append(compound)");
        append((uint8_t *)&value, sizeof(value));
    }

    ByteBuffer& operator<<(bool value)
    {
        uint8_t v = value ? 1 : 0;
        append<uint8_t>(v);
        return *this;
    }

    ByteBuffer& operator<<(uint8_t value)
    {
        append<uint8_t>(value);
        return *this;
    }

    ByteBuffer& operator<<(uint16_t value)
    {
        append<uint16_t>(value);
        return *this;
    }

    ByteBuffer& operator<<(uint32_t value)
    {
        append<uint32_t>(value);
        return *this;
    }

    ByteBuffer& operator<<(uint64_t value)
    {
        append<uint64_t>(value);
        return *this;
    }

    ByteBuffer& operator<<(int8_t value)
    {
        append<int8_t>(value);
        return *this;
    }

    ByteBuffer& operator<<(int16_t value)
    {
        append<int16_t>(value);
        return *this;
    }

    ByteBuffer& operator<<(int32_t value)
    {
        append<int32_t>(value);
        return *this;
    }

    ByteBuffer& operator<<(int64_t value)
    {
        append<int64_t>(value);
        return *this;
    }

    // floating points
    ByteBuffer& operator<<(float value)
    {
        append<float>(value);
        return *this;
    }

    ByteBuffer& operator<<(double value)
    {
        append<double>(value);
        return *this;
    }

    ByteBuffer& operator<<(long double value)
    {
        append<long double>(value);
        return *this;
    }

    ByteBuffer& operator<<(const std::string& value)
    {
        if (size_t len = value.length())
            append((const uint8_t*)value.c_str(), len);
        append((uint8_t)0);
        return *this;
    }

    ByteBuffer& operator<<(const char* str)
    {
        if (size_t len = (str ? strlen(str) : 0))
            append((const uint8_t*)str, len);
        append((uint8_t)0);
        return *this;
    }

    ByteBuffer& operator>>(bool& value)
    {
        value = read<uint8_t>() > 0 ? true : false;
        return *this;
    }

    ByteBuffer& operator>>(uint8_t& value)
    {
        value = read<uint8_t>();
        return *this;
    }

    ByteBuffer& operator>>(uint16_t& value)
    {
        value = read<uint16_t>();
        return *this;
    }

    ByteBuffer& operator>>(uint32_t& value)
    {
        value = read<uint32_t>();
        return *this;
    }

    ByteBuffer& operator>>(uint64_t& value)
    {
        value = read<uint64_t>();
        return *this;
    }

    ByteBuffer& operator>>(int8_t& value)
    {
        value = read<int8_t>();
        return *this;
    }

    ByteBuffer& operator>>(int16_t& value)
    {
        value = read<int16_t>();
        return *this;
    }

    ByteBuffer& operator>>(int32_t& value)
    {
        value = read<int32_t>();
        return *this;
    }

    ByteBuffer& operator>>(int64_t& value)
    {
        value = read<int64_t>();
        return *this;
    }

    ByteBuffer& operator>>(float& value)
    {
        value = read<float>();
        if (!std::isfinite(value))
            throw ByteBufferException();
        return *this;
    }

    ByteBuffer& operator>>(double& value)
    {
        value = read<double>();
        if (!std::isfinite(value))
            throw ByteBufferException();
        return *this;
    }

    ByteBuffer& operator>>(std::string& value)
    {
        value.clear();
        // prevent crash at wrong string format in packet
        while (rpos() < size()) {
            char c = read<char>();
            if (c == 0)
                break;
            value += c;
        }
        return *this;
    }

    uint8_t& operator[](const size_t pos)
    {
        if (pos >= size())
            throw ByteBufferPositionException(false, pos, 1, size());
        return m_storage[pos];
    }

    const uint8_t& operator[](const size_t pos) const
    {
        if (pos >= size())
            throw ByteBufferPositionException(false, pos, 1, size());
        return m_storage[pos];
    }

    size_t rpos() const { return m_rpos; }

    size_t rpos(size_t p)
    {
        m_rpos = p;
        return m_rpos;
    }

    void rfinish()
    {
        m_rpos = wpos();
    }

    size_t wpos() const { return m_wpos; }

    size_t wpos(size_t p)
    {
        m_wpos = p;
        return m_wpos;
    }

    template<typename T>
    void read_skip() { read_skip(sizeof(T)); }

    void read_skip(size_t skip)
    {
        if (m_rpos + skip > size())
            throw ByteBufferPositionException(false, m_rpos, skip, size());
        m_rpos += skip;
    }

    template <typename T> 
    T read()
    {
        T r = read<T>(m_rpos);
        m_rpos += sizeof(T);
        return r;
    }

    template <typename T> 
    T read(size_t pos) const
    {
        if (pos + sizeof(T) > size())
            throw ByteBufferPositionException(false, pos, sizeof(T), size());
        T val{};
        std::memcpy(&val, &m_storage[pos], sizeof(T));
        return val;
    }

    void read(uint8_t* dest, size_t len)
    {
        if (m_rpos  + len > size())
           throw ByteBufferPositionException(false, m_rpos, len, size());
        std::memcpy(dest, &m_storage[m_rpos], len);
        m_rpos += len;
    }

    uint8_t* contents()
    {
        if (m_storage.empty())
            throw ByteBufferException();
        return m_storage.data();
    }

    const uint8_t* contents() const
    {
        if (m_storage.empty())
            throw ByteBufferException();
        return m_storage.data();
    }

    size_t size() const { return m_storage.size(); }
    bool empty() const { return m_storage.empty(); }

    void resize(size_t newsize)
    {
        m_storage.resize(newsize, 0);
        m_rpos = 0;
        m_wpos = size();
    }

    void reserve(size_t ressize)
    {
        if (ressize > size())
            m_storage.reserve(ressize);
    }

    void append(const char* src, size_t cnt)
    {
        return append((const uint8_t*)src, cnt);
    }

    template<class T> 
    void append(const T* src, size_t cnt)
    {
        return append((const uint8_t*)src, cnt * sizeof(T));
    }

    void append(const uint8_t *src, size_t cnt)
    {
        if (!cnt)
            throw ByteBufferSourceException(m_wpos, size(), cnt);

        if (!src)
            throw ByteBufferSourceException(m_wpos, size(), cnt);

        if (m_storage.size() < m_wpos + cnt)
            m_storage.resize(m_wpos + cnt);
        std::memcpy(&m_storage[m_wpos], src, cnt);
        m_wpos += cnt;
    }

    void append(const ByteBuffer& buffer)
    {
        if (buffer.wpos())
            append(buffer.contents(), buffer.wpos());
    }

    //void textlike() const;
    //void hexlike() const;
protected:
    size_t m_rpos;
    size_t m_wpos;
    std::vector<uint8_t> m_storage;
};

template <typename T>
inline ByteBuffer& operator<<(ByteBuffer& b, const std::vector<T>& v)
{
    b << (uint32_t)v.size();
    for (typename auto i = v.begin(); i != v.end(); ++i) {
        b << *i;
    }
    return b;
}

template <typename T>
inline ByteBuffer& operator>>(ByteBuffer& b, std::vector<T> &v)
{
    uint32_t vsize;
    b >> vsize;
    v.clear();
    while (vsize--) {
        T t;
        b >> t;
        v.push_back(t);
    }
    return b;
}

template <typename T>
inline ByteBuffer& operator<<(ByteBuffer& b, const std::list<T>& v)
{
    b << (uint32_t)v.size();
    for (typename auto i = v.begin(); i != v.end(); ++i) {
        b << *i;
    }
    return b;
}

template <typename T>
inline ByteBuffer& operator>>(ByteBuffer& b, std::list<T> &v)
{
    uint32_t vsize;
    b >> vsize;
    v.clear();
    while (vsize--) {
        T t;
        b >> t;
        v.push_back(t);
    }
    return b;
}

template <typename K, typename V>
inline ByteBuffer& operator<<(ByteBuffer& b, const std::map<K, V> &m)
{
    b << (uint32_t)m.size();
    for (typename auto i = m.begin(); i != m.end(); ++i) {
        b << i->first << i->second;
    }
    return b;
}

template <typename K, typename V>
inline ByteBuffer& operator>>(ByteBuffer& b, std::map<K, V>& m)
{
    uint32_t msize;
    b >> msize;
    m.clear();
    while (msize--) {
        K k{};
        V v{};
        b >> k >> v;
        m.insert({ std::move(k), std::move(v) });
    }
    return b;
}

/*

template<> 
inline std::string ByteBuffer::read<std::string>()
{
    std::string tmp;
    *this >> tmp;
    return tmp;
}

template<>
inline void ByteBuffer::read_skip<char*>()
{
    std::string temp;
    *this >> temp;
}

template<>
inline void ByteBuffer::read_skip<char const*>()
{
    read_skip<char*>();
}

template<>
inline void ByteBuffer::read_skip<std::string>()
{
    read_skip<char*>();
}
*/

}
