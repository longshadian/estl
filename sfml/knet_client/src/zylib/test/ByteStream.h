#pragma once

#include <exception>
#include <list>
#include <map>
#include <string>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <cassert>
#include <iostream>

class ByteStreamException : public std::runtime_error
{
public:
    explicit ByteStreamException(const std::string& str) : std::runtime_error(str) { }
    //~ByteStreamException() throw() {}
};

class ByteStreamIsFinite : public ByteStreamException
{
public:
    explicit ByteStreamIsFinite(const std::string& str) : ByteStreamException(str) {}
};

class ByteStreamOverflow : public ByteStreamException
{
public:
    ByteStreamOverflow(size_t pos, size_t size, size_t remian_size);
    //~ByteStreamOverflow() throw() { }

private:
    static std::string toString(size_t pos, size_t size, size_t remain_size);
};

class ByteStream
{
public:
    static const size_t DEFAULT_SIZE = 1024;

    ByteStream();
    explicit ByteStream(size_t res);
    explicit ByteStream(std::vector<uint8_t> buf);
    explicit ByteStream(const void* src, size_t len);
    ~ByteStream();

    ByteStream(const ByteStream& right);
    ByteStream& operator=(const ByteStream& right);
    ByteStream(ByteStream&& buf);
    ByteStream& operator=(ByteStream&& right);

    void        Clear();
    void        ShrinkToFit();
    size_t      ReadSize() const { return m_write_pos - m_read_pos; }
    bool        ReadEmpty() const { return ReadSize() == 0; }

    const uint8_t* ReadData() const
    {
        if (m_storage.empty())
            return nullptr;
        return &m_storage[m_read_pos];
    }

private:
    size_t      WriteRemainSize() const { return m_storage.size() - m_write_pos; }

    void        Resize(size_t s)
    {
        m_storage.resize(s, 0);
        m_read_pos = 0;
        m_write_pos = m_storage.size();
    }

public:

    template <typename T>
    void Write(T value)
    {
        static_assert(std::is_integral<T>::value, "T must be integral");
        Write(&value, sizeof(value));
    }

    void Write(bool value)
    {
        uint8_t temp = value ? 1 : 0;
        Write(temp);
    }

    void Write(float value)
    {
        Write(&value, sizeof(value));
    }

    void Write(double value)
    {
        Write(&value, sizeof(value));
    }

    void Write(long double value)
    {
        Write(&value, sizeof(value));
    }

    void Write(const std::string& str) 
    {
        uint32_t len = static_cast<uint32_t>(str.length());
        Write(len);
        Write(str.data(), static_cast<size_t>(len));
    }

    void Write(const void* src, size_t len)
    {
        if (len == 0)
            return;
        if (!src)
            return;
        if (m_storage.size() < m_write_pos + len)
            m_storage.resize(m_write_pos + len);
        std::memcpy(&m_storage[m_write_pos], src, len);
        WriteCommit(len);
    }

private:
    void WriteCommit(size_t len)
    {
        m_write_pos += len;
    }

public:

    template <typename T> 
    void Read(T* value)
    {
        static_assert(std::is_integral<T>::value, "T must be integral");
        Read(value, sizeof(T));
    }

    void Read(bool* value)
    {
        uint8_t temp{};
        Read(&temp, sizeof(temp));
        *value = (temp != 0);
    }

    void Read(float* value)
    {
        Read(value, sizeof(float));
        if (!std::isfinite(*value))
            throw ByteStreamIsFinite("read float isfinite: false");
    }

    void Read(double* value)
    {
        Read(value, sizeof(double));
        if (!std::isfinite(*value))
            throw ByteStreamIsFinite("read double isfinite: false");
    }

    void Read(long double* value)
    {
        Read(value, sizeof(long double));
        if (!std::isfinite(*value))
            throw ByteStreamIsFinite("read long double isfinite == false");
    }

    void Read(std::string* value)
    {
        uint32_t len{};
        Read(&len);
        if (len == 0)
            return;
        size_t pos = value->size();
        value->resize(pos + len);
        Read(&(*value)[pos], len);
    }

    void Read(void* dest, size_t len)
    {
        if (len == 0)
            return;
        if (len > ReadSize())
            throw ByteStreamOverflow(m_read_pos, len, ReadSize());
        std::memcpy(dest, &m_storage[m_read_pos], len);
        ReadConsume(len);
    }

private:
    void ReadConsume(size_t len)
    {
        if (len > ReadSize())
            throw ByteStreamOverflow(m_read_pos, len, ReadSize());
        m_read_pos += len;
    }

public:
    /*
    template <typename T>
    ByteStream& operator<<(T value)
    {
        static_assert(std::is_arithmetic<T>::value, "T must be arithmetic");
        Write(&value, sizeof(value));
        WriteCommit(sizeof(value));
        return *this;
    }

    ByteStream& operator<<(bool value)
    {
        uint8_t v = value ? 1 : 0;
        (*this) << v;
        return *this;
    }

    ByteStream& operator<<(const std::string& value)
    {
        if (!value.empty()) {
            size_t len = value.length();
            Write(value.c_str(), len);
            WriteCommit(len);
        }
        return *this;
    }

    ByteStream& operator<<(const char* str)
    {
        size_t len = std::strlen(str);
        if (len != 0) {
            Write(str, len);
            WriteCommit(len);
        }
        return *this;
    }

    template <typename T>
    ByteStream& operator>>(T& value)
    {
        static_assert(std::is_integral<T>::value, "T must be integral");
        Read(&value);
        ReadConsume(sizeof(value));
        return *this;
    }

    ByteStream& operator>>(std::string& val)
    {
        val.clear();
        while (!byteEmpty()) {
            char c{};
            Read(&c); readSkip<char>();
            val.push_back(c);
        }
        return *this;
    }

    ByteStream& operator>>(std::vector<uint8_t>& val)
    {
        val.clear();
        if (!byteEmpty()) {
            val.assign(data(), data() + byteSize());
            ReadConsume(val.size());
        }
        return *this;
    }
    */
protected:
    size_t m_read_pos;
    size_t m_write_pos;
    std::vector<uint8_t> m_storage;
};
