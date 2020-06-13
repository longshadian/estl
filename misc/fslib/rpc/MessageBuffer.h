#pragma once

#include <vector>

namespace fslib {
namespace grpc {

class MessageBuffer
{
    enum { DEFAULT_STOREAGE_SIZE  =  4096 };
public:
    typedef std::vector<uint8_t>::size_type size_type;
public:

                            MessageBuffer();
    explicit                MessageBuffer(size_t initial_size);
                            MessageBuffer(const MessageBuffer& rhs);
                            MessageBuffer(MessageBuffer&& rhs);
    MessageBuffer&          operator=(const MessageBuffer& rhs);
    MessageBuffer&          operator=(MessageBuffer&& rhs);

    void                    assign(const void* p, size_type bytes);
    void                    reset();
    void                    resize(size_type bytes);
    uint8_t*                getBasePtr();
    const uint8_t*          getBasePtr() const;
    uint8_t*                getReadPtr();
    const uint8_t*          getReadPtr() const;
    uint8_t*                getWritePtr();
    const uint8_t*          getWritePtr() const;
    void                    readCompleted(size_type bytes);
    void                    writeCompleted(size_type bytes);
    size_type               getActiveSize() const;
    size_type               getBufferSize() const;
    size_type               getRemainingSpace() const;
    void                    normalize();
    void                    write(void const* data, size_t size);
    std::vector<uint8_t>&&  move();
private:
    size_type               m_wpos;
    size_type               m_rpos;
    std::vector<uint8_t>    m_storage;
};

inline
MessageBuffer::MessageBuffer() : m_wpos(0), m_rpos(0), m_storage()
{
    m_storage.resize(DEFAULT_STOREAGE_SIZE);
}

inline
MessageBuffer::MessageBuffer(size_t initial_size) : m_wpos(0), m_rpos(0), m_storage()
{
    m_storage.resize(initial_size);
}

inline
MessageBuffer::MessageBuffer(const MessageBuffer& rhs) : m_wpos(rhs.m_wpos), m_rpos(rhs.m_rpos), m_storage(rhs.m_storage)
{
}

inline
MessageBuffer::MessageBuffer(MessageBuffer&& rhs) : m_wpos(rhs.m_wpos), m_rpos(rhs.m_rpos), m_storage(rhs.move()) 
{
}

inline
MessageBuffer& MessageBuffer::operator=(const MessageBuffer& rhs)
{
    if (this != &rhs) {
        m_wpos = rhs.m_wpos;
        m_rpos = rhs.m_rpos;
        m_storage = rhs.m_storage;
    }
    return *this;
}

inline
MessageBuffer& MessageBuffer::operator=(MessageBuffer&& rhs)
{
    if (this != &rhs) {
        m_wpos = rhs.m_wpos;
        m_rpos = rhs.m_rpos;
        m_storage = rhs.move();
    }
    return *this;
}

inline
void MessageBuffer::assign(const void* p, size_type bytes)
{
    this->resize(bytes);
    write(p, bytes);
}

inline 
void MessageBuffer::reset()
{
    m_wpos = 0;
    m_rpos = 0;
}

inline
void MessageBuffer::resize(size_type bytes)
{
    m_storage.resize(bytes);
}

inline
uint8_t* MessageBuffer::getBasePtr() 
{ 
    return m_storage.data();
}

inline
const uint8_t* MessageBuffer::getBasePtr() const
{ 
    return m_storage.data();
}

inline
uint8_t* MessageBuffer::getReadPtr() 
{ 
    return &m_storage[m_rpos];
}

inline
const uint8_t* MessageBuffer::getReadPtr() const
{ 
    return &m_storage[m_rpos];
}

inline
uint8_t* MessageBuffer::getWritePtr() 
{ 
    return &m_storage[m_wpos];
}

inline
const uint8_t* MessageBuffer::getWritePtr() const
{ 
    return &m_storage[m_wpos];
}

inline 
void MessageBuffer::readCompleted(size_type bytes) 
{ 
    m_rpos += bytes;
}

inline
void MessageBuffer::writeCompleted(size_type bytes) 
{ 
    m_wpos += bytes; 
}

inline
MessageBuffer::size_type MessageBuffer::getActiveSize() const 
{ 
    return m_wpos - m_rpos; 
}

inline
MessageBuffer::size_type MessageBuffer::getRemainingSpace() const 
{
    return m_storage.size() - m_wpos; 
}

inline
MessageBuffer::size_type MessageBuffer::getBufferSize() const 
{ 
    return m_storage.size(); 
}

inline
void MessageBuffer::normalize()
{
    if (m_rpos) {
        if (m_rpos != m_wpos)
            memmove(getBasePtr(), getReadPtr(), getActiveSize());
        m_wpos -= m_rpos;
        m_rpos = 0;
    }
}

inline
void MessageBuffer::write(const void* data, size_t size)
{
    if (size) {
        memcpy(getWritePtr(), data, size);
        writeCompleted(size);
    }
}

inline
std::vector<uint8_t>&& MessageBuffer::move()
{
    m_wpos = 0;
    m_rpos = 0;
    return std::move(m_storage);
}

}
}
