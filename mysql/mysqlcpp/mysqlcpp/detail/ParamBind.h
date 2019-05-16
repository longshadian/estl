#pragma once

#include <mysql.h>

#include <vector>
#include <memory>

#include "mysqlcpp/Types.h"

namespace mysqlcpp {

class DateTime;

namespace detail {


class MYSQLCPP_EXPORT ParamBind
{
public:
    ParamBind(unsigned int paramCount);
    ~ParamBind();

    ParamBind(const ParamBind& rhs) = delete;
    ParamBind& operator=(const ParamBind& rhs) = delete;

    void setBool(uint32 index, bool value);
    void setUInt8(uint32 index, uint8 value);
    void setUInt16(uint32 index, uint16 value);
    void setUInt32(uint32 index, uint32 value);
    void setUInt64(uint32 index, uint64 value);
    void setInt8(uint32 index, int8 value);
    void setInt16(uint32 index, int16 value);
    void setInt32(uint32 index, int32 value);
    void setInt64(uint32 index, int64 value);
    void setFloat(uint32 index, float value);
    void setDouble(uint32 index, double value);
    void setString(uint32 index, const std::string& value);
    void setString(uint32 index, const char* value);
    void setBinary(uint32 index, std::vector<uint8> value, bool isString);
    void setNull(uint32 index);
    void setDateTime(uint32 index, const DateTime& value);

    MYSQL_BIND* getMYSQL_BIND();
    void clear();
private:
    void set(uint32_t index);
    void setWholeNumber(uint32 index, MYSQL_BIND* param, enum_field_types type, const void* src, uint32 src_len, bool is_unsigned);
    void setRealNumber(uint32 index, MYSQL_BIND* param, enum_field_types type, const void* src, uint32 src_len);
    bool checkValidIndex(uint32 index);
private:
    unsigned int                    m_param_count;
    std::vector<MYSQL_BIND>         m_bind;
    std::vector<bool>               m_value_set;
    std::vector<std::vector<uint8_t>> m_bind_buffer;
};

} //detail

}
