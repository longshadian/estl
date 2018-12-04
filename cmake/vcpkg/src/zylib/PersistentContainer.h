#pragma once

#include <cstring>
#include <vector>
#include <list>
#include <type_traits>

namespace zylib {
namespace detail {
template <typename T> struct PstTool;
}
}

namespace zylib {

template <typename T>
struct PersistentContainer
{
    using ContainerType = T;
    using ValueType = typename ContainerType::value_type;

    enum { SENTINEL_SIZE = sizeof(int32_t) };
    enum { VALUE_SIZE = sizeof(ValueType) };

    static_assert(std::is_pod<ValueType>::value,    "T mush be POD!");
    static_assert(VALUE_SIZE > 0,                   "sizeof(T) mush > 0!");

    PersistentContainer() 
        : m_null_sentinel()
        , m_data() 
    {
    }

    ~PersistentContainer() 
    {
    }

    PersistentContainer(const PersistentContainer& rhs) 
        : m_null_sentinel(rhs.m_null_sentinel) 
        , m_data(rhs.m_data) 
    {
    }

    PersistentContainer& operator=(const PersistentContainer& rhs)
    {
        if (this != &rhs) {
            m_null_sentinel = rhs.m_null_sentinel;
            m_data = rhs.m_data;
        }
        return *this;
    }

    PersistentContainer(PersistentContainer&& rhs) 
        : m_null_sentinel(rhs.m_null_sentinel) 
        , m_data(std::move(rhs.m_data)) 
    {
    }

    PersistentContainer& operator=(PersistentContainer&& rhs)
    {
        if (this != &rhs) {
            m_null_sentinel = rhs.m_null_sentinel;
            m_data = std::move(rhs.m_data);
        }
        return *this;
    }

    bool operator==(const PersistentContainer& rhs) const
    {
        return m_null_sentinel == rhs.m_null_sentinel
            &&  m_data == rhs.m_data;
    }

    bool operator!=(const PersistentContainer& rhs) const
    {
        return !(*this == rhs);
    }

    size_t bytes() const
    {
        return SENTINEL_SIZE + VALUE_SIZE * m_data.size();
    }

    std::vector<uint8_t> serializeToBinary() const
    {
        std::vector<uint8_t> buffer{};
        serializeToBinary(&buffer);
        return buffer;
    }

    void serializeToBinary(std::vector<uint8_t>* buffer) const
    {
        detail::PstTool<ValueType>::toBinary(*this, buffer);
    }

    bool parseFromBinary(const std::vector<uint8_t>& data)
    {
        if (data.empty())
            return false;
        return parseFromBinary(data.data(), data.size());
    }

    bool parseFromBinary(const uint8_t* pos, size_t len)
    {
        return detail::PstTool<ValueType>::fromBinary(pos, len, this);
    }

    int32_t         m_null_sentinel; // 空值哨兵，确保数据为空时，序列化后大小不等于0，至少>=4
    ContainerType   m_data;
};

namespace detail {

template <typename T>
struct PstTool
{
    // vector
    static void toBinary(
        const PersistentContainer<std::vector<T>>& container
        , std::vector<uint8_t>* buffer)
    {
        using ContainerType = PersistentContainer<std::vector<T>>;

        buffer->clear();
        buffer->resize(container.bytes());
        auto* pos = buffer->data();
        std::memcpy(pos, &container.m_null_sentinel, ContainerType::SENTINEL_SIZE);
        pos += ContainerType::SENTINEL_SIZE;
        if (container.m_data.empty())
            return;
        std::memcpy(pos, container.m_data.data(), ContainerType::VALUE_SIZE * container.m_data.size());
    }

    static bool fromBinary(const uint8_t* pos, size_t len, PersistentContainer<std::vector<T>>* container)
    {
        using ContainerType = PersistentContainer<std::vector<T>>;
        if (len < ContainerType::SENTINEL_SIZE)
            return false;
        auto data_len = len - ContainerType::SENTINEL_SIZE;
        if ((data_len % ContainerType::VALUE_SIZE) != 0)
            return false;
        auto cnt = data_len / ContainerType::VALUE_SIZE;

        std::memcpy(&container->m_null_sentinel, pos, ContainerType::SENTINEL_SIZE);
        pos += ContainerType::SENTINEL_SIZE;

        if (cnt != 0) {
            container->m_data.resize(cnt);
            std::memcpy(container->m_data.data(), pos, data_len);
        }
        return true;
    }


    // list
    static void toBinary(
        const PersistentContainer<std::list<T>>& container
        , std::vector<uint8_t>* buffer)
    {
        using ContainerType = PersistentContainer<std::vector<T>>;

        buffer->clear();
        buffer->resize(container.bytes());
        auto* pos = buffer->data();
        std::memcpy(pos, &container.m_null_sentinel, ContainerType::SENTINEL_SIZE);
        pos += ContainerType::SENTINEL_SIZE;
        if (container.m_data.empty())
            return;

        for (const auto& slot : container.m_data) {
            std::memcpy(pos, &slot, ContainerType::VALUE_SIZE);
            pos += ContainerType::VALUE_SIZE;
        }
    }

    static bool fromBinary(const uint8_t* pos, size_t len, PersistentContainer<std::list<T>>* container)
    {
        using ContainerType = PersistentContainer<std::vector<T>>;

        if (len < ContainerType::SENTINEL_SIZE)
            return false;
        auto data_len = len - ContainerType::SENTINEL_SIZE;
        if ((data_len % ContainerType::VALUE_SIZE) != 0)
            return false;
        auto cnt = data_len / ContainerType::VALUE_SIZE;

        std::memcpy(&container->m_null_sentinel, pos, ContainerType::SENTINEL_SIZE);
        pos += ContainerType::SENTINEL_SIZE;

        while (cnt > 0) {
            --cnt;
            T slot{};
            std::memcpy(&slot, pos, ContainerType::VALUE_SIZE);
            pos += ContainerType::VALUE_SIZE;
            container->m_data.emplace_back(slot);
        }
        return true;
    }

};

} // detail
} // zylib
