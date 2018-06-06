#include "ByteBuffer.h"

#include <cstring>

BinaryByteBuffer::BinaryByteBuffer()
    : m_storage()
    , m_rpos()
    , m_wpos()
{
}

BinaryByteBuffer::BinaryByteBuffer(std::vector<uint8_t> data)
    : m_storage(std::move(data))
    , m_rpos()
    , m_wpos(m_storage.size())
{
}

BinaryByteBuffer::BinaryByteBuffer(const std::string& data)
    : m_storage(data.begin(), data.end())
    , m_rpos()
    , m_wpos()
{
}

BinaryByteBuffer::BinaryByteBuffer(const void* data, size_t len)
    : m_storage(reinterpret_cast<const uint8_t*>(data), reinterpret_cast<const uint8_t*>(data) + len)
    , m_rpos()
    , m_wpos()
{
}

size_t BinaryByteBuffer::Capacity() const
{
    return m_storage.size();
}

void BinaryByteBuffer::Resize(size_t len)
{
    if (len < WriterIndex())
        return;
    m_storage.resize(len);
}

void BinaryByteBuffer::Append(const void* p, size_t len)
{
    if (len == 0) 
        return;
    if (!p)
        return;
    if (m_storage.size() < m_wpos + len)
        m_storage.resize(m_wpos + len);
    std::memcpy(&m_storage[m_wpos], p, len);
    m_wpos += len;
}

void BinaryByteBuffer::Clear()
{
    m_rpos = 0;
    m_wpos = 0;
}

void BinaryByteBuffer::ShrinkToFit()
{
    m_storage.resize(m_wpos);
}

void BinaryByteBuffer::MemoryMove()
{
    if (m_rpos == 0)
        return;
    size_t len = ReadableSize();
    if (len == 0) {
        m_rpos = 0;
        m_wpos = 0;
        return;
    }

    std::memmove(m_storage.data(), &m_storage[m_rpos], len);
    m_rpos = 0;
    m_wpos = 0;
}

size_t BinaryByteBuffer::ReaderIndex() const
{
    return m_rpos;
}

size_t BinaryByteBuffer::ReadableSize() const
{
    return WriterIndex() - ReaderIndex();
}

const void* BinaryByteBuffer::GetReaderPtr() const
{
    return &m_storage[m_rpos];
}

void BinaryByteBuffer::ReaderPickup(size_t len)
{
    m_rpos += len;
}

size_t BinaryByteBuffer::WriterIndex() const
{
    return m_wpos;
}

size_t BinaryByteBuffer::WritableSize() const
{
    return m_storage.size() - WriterIndex();
}
