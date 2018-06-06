#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace network {

class Message
{
public:
    using StorageType = std::vector<uint8_t>;
public:
    explicit Message(const std::string& data);
    explicit Message(const void* data, size_t len);
    explicit Message(StorageType data);
    ~Message() = default;

    Message(const Message& rhs);
    Message& operator=(const Message& rhs);

    Message(Message&& rhs);
    Message& operator=(Message&& rhs);

public:

    const void* data() const;
    size_t size() const;
private:
    StorageType m_data;
};

}
