#include "Message.h"

#include <cstring>
#include <algorithm>

namespace network {

Message::Message(const std::string& data)
    :Message(data.c_str(), data.length())
{
}

Message::Message(const void* data, size_t len)
    : m_data()
{
    m_data.resize(len);
    std::memcpy(m_data.data(), data, len);
}

Message::Message(StorageType data)
    : m_data(std::move(data))
{
}

Message::Message(const Message& rhs)
    : m_data(rhs.m_data)
{
}

Message& Message::operator=(const Message& rhs)
{
    if (this != &rhs) {
        m_data = rhs.m_data;
    }
    return *this;
}

Message::Message(Message&& rhs)
    : m_data(std::move(rhs.m_data))
{
}

Message& Message::operator=(Message&& rhs)
{
    if (this != &rhs) {
        m_data = std::move(rhs.m_data);
    }
    return *this;
}

const void* Message::data() const
{
    return m_data.data();
}

size_t Message::size() const
{
    return m_data.size();
}

}

