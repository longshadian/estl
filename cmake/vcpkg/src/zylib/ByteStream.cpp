#include "zylib/ByteStream.h"

#include <sstream>
#include <iostream>

namespace zylib {

ByteStreamOverflow::ByteStreamOverflow(size_t pos, size_t size, size_t remian_size)
    : ByteStreamException(ToString(pos, size, remian_size))
{
}

std::string ByteStreamOverflow::ToString(size_t pos, size_t size, size_t remian_size)
{
    std::ostringstream ss;
    ss << "overflow: pos:" << pos << " size:" << size << " "
        << " in ByteBuffer remian size:" << remian_size;
    return ss.str();
}

ByteStream::ByteStream()
    : m_read_pos(0), m_write_pos(0), m_storage(DEFAULT_SIZE)
{
}

ByteStream::ByteStream(size_t res)
    : m_read_pos(0), m_write_pos(0), m_storage(res)
{
}

ByteStream::ByteStream(std::vector<uint8_t> buf)
    : m_read_pos(0), m_write_pos(0), m_storage(std::move(buf))
{
    m_write_pos = m_storage.size();
}

ByteStream::ByteStream(const void* src, size_t len) 
    : m_read_pos(0), m_write_pos(0), m_storage(static_cast<const uint8_t*>(src), static_cast<const uint8_t*>(src) + len) 
{
    m_write_pos = m_storage.size();
}

ByteStream::~ByteStream() 
{
}

ByteStream::ByteStream(ByteStream&& buf)
    : m_read_pos(buf.m_read_pos), m_write_pos(buf.m_write_pos), m_storage(std::move(buf.m_storage))
{
}

ByteStream::ByteStream(const ByteStream& right)
    : m_read_pos(right.m_read_pos), m_write_pos(right.m_write_pos), m_storage(right.m_storage)
{
}

ByteStream& ByteStream::operator=(const ByteStream& right)
{
    if (this != &right) {
        m_read_pos = right.m_read_pos;
        m_write_pos = right.m_write_pos;
        m_storage = right.m_storage;
    }
    return *this;
}

ByteStream& ByteStream::operator=(ByteStream&& right) 
{
    if (this != &right) {
        m_read_pos = right.m_read_pos;
        m_write_pos = right.m_write_pos;
        std::swap(m_storage, right.m_storage);
    }
    return *this;
}

void ByteStream::Clear()
{
    m_read_pos = 0;
    m_write_pos = 0;
    m_storage.clear();
}

void ByteStream::ShrinkToFit()
{
    if (m_read_pos == 0)
        return;
    auto read_size = ReadSize();
    if (read_size == 0) {
        m_read_pos = 0;
        m_write_pos = 0;
        m_storage.shrink_to_fit();
        return;
    }
    std::memmove(m_storage.data(), &m_storage[m_read_pos], read_size);
    m_read_pos = 0;
    m_write_pos = read_size;
    m_storage.resize(m_write_pos);
}

} // zylib
