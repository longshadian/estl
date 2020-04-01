#pragma once

#include <cstddef>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>
#include <array>
#include <string_view>

namespace zylib {
namespace detail {

inline
std::string concatDetail(std::ostringstream* ostm)
{
    return ostm->str();
}

template <typename T, typename... Args>
inline
std::string concatDetail(std::ostringstream* ostm, T&& t, Args&&... arg)
{
    (*ostm) << std::forward<T>(t);
    return concatDetail(ostm, std::forward<Args>(arg)...);
}

} /// detail


template <typename... Args>
inline
std::string concat(Args&&... arg)
{
    std::ostringstream ostm;
    return detail::concatDetail(&ostm, std::forward<Args&&>(arg)...);
}

template <size_t L>
struct FixedString
{
    using ClassType = FixedString;
    using SizeType = size_t;

    enum { MAX_LEN = L };
    std::array<char, L>     m_data;
    SizeType                m_length;

    std::string_view GetString() const
    {
        return std::string_view(m_data.data(), m_length);
    }

    bool SetString(std::string_view sv)
    {
        if (sv.size() > MAX_LEN)
            return false;
        m_data.fill(0);
        std::copy(sv.begin(), sv.end(), m_data.begin());
        m_length = sv.size();
        return true;
    }

    size_t Length() const
    {
        return m_length;
    }

    std::vector<std::byte> AsBinary() const
    {
        std::vector<std::byte> buffer{};
        buffer.resize(Length());
        std::memcpy(buffer.data(), m_data.data(), Length());
        return buffer;
    }

    bool SetBinary(const std::vector<std::byte>& buffer)
    {
        if (buffer.size() > MAX_LEN)
            return false;
        m_data.fill(0);
        std::memcpy(m_data.data(), buffer.data(), buffer.size());
        m_length = buffer.size();
        return true;
    }

    bool operator==(const ClassType& rhs) const
    {
        return m_data == rhs.m_data;
    }

    bool operator!=(const ClassType& rhs) const
    {
        return !(*this == rhs);
    }

    friend std::ostream& operator<<(std::ostream& ostm, const ClassType& rhs)
    {
        ostm << rhs.GetString();
        return ostm;
    }
};

static_assert(std::is_pod<FixedString<0>>::value, "FixedString must be POD!");

} /// zylib
