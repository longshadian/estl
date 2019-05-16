#include "mysqlcpp/detail/ParamBind.h"

#include <mysql.h>

#include <cstring>
#include <vector>
#include <memory>

#include "mysqlcpp/MysqlcppLog.h"
#include "mysqlcpp/DateTime.h"
#include "mysqlcpp/MysqlcppAssert.h"

namespace mysqlcpp {

namespace detail {

ParamBind::ParamBind(unsigned int param_count)
    : m_param_count(param_count)
    , m_bind()
    , m_value_set()
    , m_bind_buffer()
{
    if (m_param_count > 0) {
        m_bind.resize(m_param_count);
        std::memset(m_bind.data(), 0, sizeof(MYSQL_BIND) * m_param_count);
        m_value_set.resize(m_param_count);
        m_bind_buffer.resize(m_param_count);
    }
}

ParamBind::~ParamBind()
{
}

void ParamBind::setBool(uint32 index, bool value)
{
    setUInt8(index, value ? 1 : 0);
}

void ParamBind::setUInt8(uint32 index, uint8 value)
{
    checkValidIndex(index);
    set(index);
    MYSQL_BIND* param = &m_bind[index];
    setWholeNumber(index, param, MYSQL_TYPE_TINY, &value, sizeof(uint8), true);
}

void ParamBind::setUInt16(uint32 index, uint16 value)
{
    checkValidIndex(index);
    set(index);
    MYSQL_BIND* param = &m_bind[index];
    setWholeNumber(index, param, MYSQL_TYPE_SHORT, &value, sizeof(uint16), true);
}

void ParamBind::setUInt32(uint32 index, uint32 value)
{
    checkValidIndex(index);
    set(index);
    MYSQL_BIND* param = &m_bind[index];
    setWholeNumber(index, param, MYSQL_TYPE_LONG, &value, sizeof(uint32), true);
}

void ParamBind::setUInt64(uint32 index, uint64 value)
{
    checkValidIndex(index);
    set(index);
    MYSQL_BIND* param = &m_bind[index];
    setWholeNumber(index, param, MYSQL_TYPE_LONGLONG, &value, sizeof(uint64), true);
}

void ParamBind::setInt8(uint32 index, int8 value)
{
    checkValidIndex(index);
    set(index);
    MYSQL_BIND* param = &m_bind[index];
    setWholeNumber(index, param, MYSQL_TYPE_TINY, &value, sizeof(int8), false);
}

void ParamBind::setInt16(uint32 index, int16 value)
{
    checkValidIndex(index);
    set(index);
    MYSQL_BIND* param = &m_bind[index];
    setWholeNumber(index, param, MYSQL_TYPE_SHORT, &value, sizeof(int16), false);
}

void ParamBind::setInt32(uint32 index, int32 value)
{
    checkValidIndex(index);
    set(index);
    MYSQL_BIND* param = &m_bind[index];
    setWholeNumber(index, param, MYSQL_TYPE_LONG, &value, sizeof(int32), false);
}

void ParamBind::setInt64(uint32 index, int64 value)
{
    checkValidIndex(index);
    set(index);
    MYSQL_BIND* param = &m_bind[index];
    setWholeNumber(index, param, MYSQL_TYPE_LONGLONG, &value, sizeof(int64), false);
}

void ParamBind::setFloat(uint32 index, float value)
{
    checkValidIndex(index);
    set(index);
    MYSQL_BIND* param = &m_bind[index];
    setRealNumber(index, param, MYSQL_TYPE_FLOAT, &value, sizeof(float));
}

void ParamBind::setDouble(uint32 index, double value)
{
    checkValidIndex(index);
    set(index);
    MYSQL_BIND* param = &m_bind[index];
    setRealNumber(index, param, MYSQL_TYPE_DOUBLE, &value, sizeof(double));
}

void ParamBind::setString(uint32 index, const std::string& value)
{
    if (value.empty()) {
        setBinary(index, {}, true);
    } else {
        const uint8* pos = (const uint8*)value.c_str();
        std::vector<uint8> buffer{ pos, pos + value.size() };
        setBinary(index, std::move(buffer), true);
    }
}

void ParamBind::setString(uint32 index, const char* value)
{
    auto len = std::strlen(value);
    if (len == 0) {
        setBinary(index, {}, true);
    } else {
        const uint8* pos = (const uint8*)value;
        std::vector<uint8> buffer{ pos, pos + len };
        setBinary(index, std::move(buffer), true);
    }
}

void ParamBind::setBinary(uint32 index, std::vector<uint8> value, bool is_string)
{
    checkValidIndex(index);
    set(index);

    unsigned long value_len = static_cast<unsigned long>(value.size());
    m_bind_buffer[index] = std::move(value);
    MYSQL_BIND* param = &m_bind[index];
    if (m_bind_buffer[index].empty())
        param->buffer = nullptr;
    else
        param->buffer = m_bind_buffer[index].data();
    param->buffer_type = MYSQL_TYPE_BLOB;
    param->buffer_length = value_len;
    if (is_string) {
        param->buffer_type = MYSQL_TYPE_VAR_STRING;
    }
}

void ParamBind::setNull(uint32 index)
{
    /*
    *  my_bool *is_null
    *  This member points to a my_bool variable that is true if a value is NULL, false if it is not NULL.For
    *  input, set *is_null to true to indicate that you are passing a NULL value as a statement parameter.
    *  is_null is a pointer to a boolean scalar, not a boolean scalar, to provide flexibility in how you specify NULL values:
    *   If your data values are always NULL, use MYSQL_TYPE_NULL as the buffer_type value when
    *  you bind the column. The other MYSQL_BIND members, including is_null, do not matter.
    *   If your data values are always NOT NULL, set is_null = (my_bool*) 0, and set the other
    *  members appropriately for the variable you are binding.
    *   In all other cases, set the other members appropriately and set is_null to the address of a
    *  my_bool variable. Set that variable's value to true or false appropriately between executions to
    *  indicate whether the corresponding data value is NULL or NOT NULL, respectively.
    *  For output, when you fetch a row, MySQL sets the value pointed to by is_null to true or false
    *  according to whether the result set column value returned from the statement is or is not NULL
    */

    checkValidIndex(index);
    set(index);
    MYSQL_BIND* param = &m_bind[index];
    param->buffer_type = MYSQL_TYPE_NULL;
}

void ParamBind::setDateTime(uint32 index, const DateTime& value)
{
    checkValidIndex(index);
    set(index);

    auto buffer = value.getBinary();
    m_bind_buffer[index] = std::move(buffer);

    MYSQL_BIND* param = &m_bind[index];
    param->buffer = m_bind_buffer[index].data();
    param->buffer_type = MYSQL_TYPE_DATETIME;
}

MYSQL_BIND* ParamBind::getMYSQL_BIND()
{
    if (m_bind.empty())
        return nullptr;
    return m_bind.data();
}

void ParamBind::clear()
{
    m_bind.clear();
    m_value_set.clear();
    m_bind_buffer.clear();
    if (m_param_count > 0) {
        m_bind.resize(m_param_count);
        std::memset(m_bind.data(), 0, sizeof(MYSQL_BIND) * m_param_count);

        m_value_set.resize(m_param_count);
        m_bind_buffer.resize(m_param_count);
    }
}

void ParamBind::set(uint32_t index)
{
    m_value_set[index] = true;
}

void ParamBind::setWholeNumber(uint32 index, MYSQL_BIND* param, enum_field_types type,
    const void* src, uint32 src_len, bool is_unsigned)
{
    m_bind_buffer[index].resize(src_len);
    std::memcpy(m_bind_buffer[index].data(), src, src_len);

    param->buffer_type = type;
    param->buffer = m_bind_buffer[index].data();
    param->is_unsigned = is_unsigned ? 1 : 0;
}

void ParamBind::setRealNumber(uint32 index, MYSQL_BIND* param, enum_field_types type,
    const void* src, uint32 src_len)
{
    m_bind_buffer[index].resize(src_len);
    std::memcpy(m_bind_buffer[index].data(), src, src_len);

    param->buffer_type = type;
    param->buffer = m_bind_buffer[index].data();
}

bool ParamBind::checkValidIndex(uint32 index)
{
    MYSQLCPP_ASSERT(index < m_param_count);
    if (m_value_set[index])
        MYSQLCPP_LOG(Warning, "Prepared Statement trying to bind value on already bound index: %u", index);
    return true;
}

} // detail
} // mysqlcpp
