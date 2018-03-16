#ifndef _MYSQLCPP_SAFESTRING_H
#define _MYSQLCPP_SAGESTRING_H

#include <cstring>
#include <vector>
//#include "Types.h"

using uint8 = uint8_t;

namespace mysqlcpp {

class SafeString
{
public:
    SafeString() : m_data() , m_len()
    {
        clear();
    }

    ~SafeString()
    {
    }

    SafeString(const SafeString& rhs) : m_data(rhs.m_data) , m_len(rhs.m_len) 
    {
    }

    SafeString& operator=(const SafeString& rhs)
    {
        if (this != &rhs) {
            m_data = rhs.m_data;
            m_len = rhs.m_len;
        }
        return *this;
    }

    SafeString(SafeString&& rhs) : m_data(std::move(rhs.m_data)), m_len(rhs.m_len)
    {
        rhs.clear();
    }

    SafeString& operator=(SafeString&& rhs)
    {
        if (this != &rhs) {
            m_data = std::move(rhs.m_data);
            m_len = rhs.m_len;
            rhs.clear();
        }
        return *this;
    }

    void            resize(size_t len) { resizePulsOne(len); }
    uint8*          getPtr() { return m_data.data(); }
    const uint8*    getPtr() const { return m_data.data(); }
    const char*     getCString() const { return reinterpret_cast<const char*>(getPtr()); }
    size_t          getLength() const { return m_len; }
    size_t          getCapacity() const { return m_data.size(); }
    bool            empty() const { return m_len == 0; }
    void            clear() { resize(0); }
    std::vector<uint8> getBinary() const { return std::vector<uint8>{getPtr(), getPtr() + getLength()}; }
private:
    void resizePulsOne(size_t len)
    {
        m_len = len;
        m_data.resize(m_len + 1);
        m_data[m_len] = 0;
    }
private:
    std::vector<uint8> m_data;
    size_t             m_len;
};

}

#endif
