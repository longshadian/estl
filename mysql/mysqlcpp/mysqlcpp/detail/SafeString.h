#pragma once

#include <vector>
#include <string_view>
#include "mysqlcpp/Types.h"

namespace mysqlcpp {
namespace detail {

class MYSQLCPP_EXPORT SafeString
{
public:
    using Type = uint8;
    using Container = std::vector<Type>;
public:
    SafeString() : m_data() {}
    ~SafeString() = default;

    SafeString(const SafeString& rhs) : m_data(rhs.m_data) {}
    SafeString& operator=(const SafeString& rhs)
    {
        if (this != &rhs) {
            m_data = rhs.m_data;
        }
        return *this;
    }

    SafeString(SafeString&& rhs) : m_data(std::move(rhs.m_data)) { }
    SafeString& operator=(SafeString&& rhs)
    {
        if (this != &rhs) {
            std::swap(m_data, rhs.m_data);
        }
        return *this;
    }

    void                    Resize(size_t len);
    void                    Clear();
    bool                    Empty() const;
    Type*                   GetPtr();
    const Type*             GetPtr() const;
    size_t                  Length() const;

    std::string_view        AsStringView() const;
    void                    AppendBuffer(const void* src, size_t len);

    const Container&        AsBinary() const &;
    Container&              AsBinary() &;
    Container               AsBinary() &&;

private:
    const char*             AsCString() const;
private:
    Container               m_data;
};

inline
void SafeString::Resize(size_t len) 
{ 
    m_data.resize(len);
}

inline
void SafeString::Clear()
{
    m_data.clear();
}

inline
bool SafeString::Empty() const
{
    return m_data.empty();
}

inline
SafeString::Type* SafeString::GetPtr()
{
    return m_data.empty() ? nullptr : m_data.data();
}

inline
const SafeString::Type* SafeString::GetPtr() const
{
    return m_data.empty() ? nullptr : m_data.data();
}

inline
size_t SafeString::Length() const
{
    return m_data.size();
}

inline
std::string_view SafeString::AsStringView() const
{ 
    if (Empty())
        return std::string_view{};
    return std::string_view{ AsCString(), Length()}; 
}

inline
const char* SafeString::AsCString() const
{
    if (m_data.empty())
        return nullptr;
    return reinterpret_cast<const char*>(m_data.data());
}

inline
void SafeString::AppendBuffer(const void* src, size_t len)
{
    if (!src || len == 0)
        return;
    auto old_len = Length();
    Resize(Length() + len);
    std::memcpy(m_data.data() + old_len, src, len);
}

inline
SafeString::Container& SafeString::AsBinary() &
{
    return m_data;
}

inline
const SafeString::Container& SafeString::AsBinary() const &
{
    return m_data;
}

inline
SafeString::Container SafeString::AsBinary() &&
{
    return std::move(m_data);
}


} // detail
} // mysqlcpp
