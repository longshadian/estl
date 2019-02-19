#ifndef _MYSQLCPP_PREPAREDSTATEMENT_H
#define _MYSQLCPP_PREPAREDSTATEMENT_H

#include <mysql.h>
#include <vector>

#include "Types.h"
#include "DateTime.h"

namespace mysqlcpp {

class PreparedStatement
{
    friend class Connection;
public:
    PreparedStatement(MYSQL_STMT* stmt);
    ~PreparedStatement();
    PreparedStatement(const PreparedStatement& right) = delete;
    PreparedStatement& operator=(const PreparedStatement& right) = delete;
public:

    void setBool(const uint8 index, const bool value);
    void setUInt8(const uint8 index, const uint8 value);
    void setUInt16(const uint8 index, const uint16 value);
    void setUInt32(const uint8 index, const uint32 value);
    void setUInt64(const uint8 index, const uint64 value);
    void setInt8(const uint8 index, const int8 value);
    void setInt16(const uint8 index, const int16 value);
    void setInt32(const uint8 index, const int32 value);
    void setInt64(const uint8 index, const int64 value);
    void setFloat(const uint8 index, const float value);
    void setDouble(const uint8 index, const double value);
    void setString(const uint8 index, const std::string& value);
    void setString(const uint8 index, const char* value);
    void setBinary(const uint8 index, std::vector<uint8> value, bool isString);
    void setNull(const uint8 index);
    void setDateTime(const uint8 index, const DateTime value);

    void clearParameters();
    MYSQL_STMT* getMYSQL_STMT();
    MYSQL_BIND* getMYSQL_BIND();
private:
    bool checkValidIndex(uint8 index);
    //std::string getQueryString(std::string const& sqlPattern) const;
    void setWholeNumber(const uint8 index, MYSQL_BIND* param, enum_field_types type, const void* src, uint32 src_len, bool is_unsigned);
    void setRealNumber(const uint8 index, MYSQL_BIND* param, enum_field_types type, const void* src, uint32 src_len);
private:
    MYSQL_STMT*             m_stmt;
    std::vector<MYSQL_BIND> m_bind_param;
    std::vector<bool>       m_param_set;
    uint32_t                m_param_count;
    std::vector<std::vector<uint8>> m_bind_param_buffer;
};

}

#endif
