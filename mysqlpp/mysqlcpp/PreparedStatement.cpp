#include "PreparedStatement.h"

#include <cstring>

#include "Assert.h"
#include "Connection.h"
#include "FakeLog.h"

namespace mysqlcpp {

PreparedStatement::PreparedStatement(MYSQL_STMT* stmt) 
    : m_stmt(stmt)
    , m_bind_param()
    , m_param_set()
    , m_param_count()
    , m_bind_param_buffer()
{
    m_param_count = ::mysql_stmt_param_count(m_stmt);
    m_param_set.resize(m_param_count);
    m_bind_param.resize(m_param_count);
    std::memset(m_bind_param.data(), 0, sizeof(MYSQL_BIND) * m_bind_param.size());

    m_bind_param_buffer.resize(m_param_count);

    /// "If set to 1, causes mysql_stmt_store_result() to update the metadata MYSQL_FIELD->max_length value."
    my_bool bool_tmp = 1;
    ::mysql_stmt_attr_set(stmt, STMT_ATTR_UPDATE_MAX_LENGTH, &bool_tmp);
}

PreparedStatement::~PreparedStatement()
{
    if (m_stmt)
        ::mysql_stmt_close(m_stmt);
}

void PreparedStatement::clearParameters()
{
    m_param_set.clear();
    m_param_set.resize(m_param_count);

    m_bind_param.clear();
    m_bind_param.resize(m_param_count);
    std::memset(m_bind_param.data(), 0, sizeof(MYSQL_BIND) * m_bind_param.size());

    m_bind_param_buffer.clear();
    m_bind_param_buffer.resize(m_param_count);
}

MYSQL_STMT* PreparedStatement::getMYSQL_STMT() 
{ 
    return m_stmt; 
}

MYSQL_BIND* PreparedStatement::getMYSQL_BIND() 
{ 
    if (m_bind_param.empty())
        return nullptr;
    return m_bind_param.data();
}

bool PreparedStatement::checkValidIndex(uint8 index)
{
    ASSERT(index < m_param_count);
    if (m_param_set[index]) FAKE_LOG_ERROR() << "[WARNING] Prepared Statement trying to bind value on already bound index " << index;
    return true;
}

void PreparedStatement::setBool(const uint8 index, const bool value)
{
    setUInt8(index, value ? 1 : 0);
}

void PreparedStatement::setUInt8(const uint8 index, const uint8 value)
{
    checkValidIndex(index);
    m_param_set[index] = true;
    MYSQL_BIND* param = &m_bind_param[index];
    setWholeNumber(index, param, MYSQL_TYPE_TINY, &value, sizeof(uint8), true);
}

void PreparedStatement::setUInt16(const uint8 index, const uint16 value)
{
    checkValidIndex(index);
    m_param_set[index] = true;
    MYSQL_BIND* param = &m_bind_param[index];
    setWholeNumber(index, param, MYSQL_TYPE_SHORT, &value, sizeof(uint16), true);
}

void PreparedStatement::setUInt32(const uint8 index, const uint32 value)
{
    checkValidIndex(index);
    m_param_set[index] = true;
    MYSQL_BIND* param = &m_bind_param[index];
    setWholeNumber(index, param, MYSQL_TYPE_LONG, &value, sizeof(uint32), true);
}

void PreparedStatement::setUInt64(const uint8 index, const uint64 value)
{
    checkValidIndex(index);
    m_param_set[index] = true;
    MYSQL_BIND* param = &m_bind_param[index];
    setWholeNumber(index, param, MYSQL_TYPE_LONGLONG, &value, sizeof(uint64), true);
}

void PreparedStatement::setInt8(const uint8 index, const int8 value)
{
    checkValidIndex(index);
    m_param_set[index] = true;
    MYSQL_BIND* param = &m_bind_param[index];
    setWholeNumber(index, param, MYSQL_TYPE_TINY, &value, sizeof(int8), false);
}

void PreparedStatement::setInt16(const uint8 index, const int16 value)
{
    checkValidIndex(index);
    m_param_set[index] = true;
    MYSQL_BIND* param = &m_bind_param[index];
    setWholeNumber(index, param, MYSQL_TYPE_SHORT, &value, sizeof(int16), false);
}

void PreparedStatement::setInt32(const uint8 index, const int32 value)
{
    checkValidIndex(index);
    m_param_set[index] = true;
    MYSQL_BIND* param = &m_bind_param[index];
    setWholeNumber(index, param, MYSQL_TYPE_LONG, &value, sizeof(int32), false);
}

void PreparedStatement::setInt64(const uint8 index, const int64 value)
{
    checkValidIndex(index);
    m_param_set[index] = true;
    MYSQL_BIND* param = &m_bind_param[index];
    setWholeNumber(index, param, MYSQL_TYPE_LONGLONG, &value, sizeof(int64), false);
}

void PreparedStatement::setFloat(const uint8 index, const float value)
{
    checkValidIndex(index);
    m_param_set[index] = true;
    MYSQL_BIND* param = &m_bind_param[index];
    setRealNumber(index, param, MYSQL_TYPE_FLOAT, &value, sizeof(float));
}

void PreparedStatement::setDouble(const uint8 index, const double value)
{
    checkValidIndex(index);
    m_param_set[index] = true;
    MYSQL_BIND* param = &m_bind_param[index];
    setRealNumber(index, param, MYSQL_TYPE_DOUBLE, &value, sizeof(double));
}

void PreparedStatement::setString(const uint8 index, const std::string& value)
{
    const uint8* pos = (const uint8*)value.data();
    std::vector<uint8> buffer{pos, pos + value.size()};
    setBinary(index, std::move(buffer), true);
}

void PreparedStatement::setString(const uint8 index, const char* value)
{
    const uint8* pos = (const uint8*)value;
    auto len = std::strlen(value);
    std::vector<uint8> buffer{pos, pos + len};
    setBinary(index, std::move(buffer), true);
}

void PreparedStatement::setBinary(const uint8 index, std::vector<uint8> src, bool is_string)
{
    checkValidIndex(index);
    m_param_set[index] = true;

    auto src_len = src.size();
    m_bind_param_buffer[index] = std::move(src);

    MYSQL_BIND* param = &m_bind_param[index];
    param->buffer = m_bind_param_buffer[index].data(); 
    param->buffer_type = MYSQL_TYPE_BLOB;
    param->buffer_length = (unsigned long)src_len;
    if (is_string) {
        param->buffer_type = MYSQL_TYPE_VAR_STRING;
    }
}

void PreparedStatement::setNull(const uint8 index)
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
    m_param_set[index] = true;

    MYSQL_BIND* param = &m_bind_param[index];
    param->buffer_type = MYSQL_TYPE_NULL;
}

void PreparedStatement::setDateTime(const uint8 index, const DateTime tm)
{
    checkValidIndex(index);
    m_param_set[index] = true;

    auto buffer = tm.getBinary();
    //auto src_len = buffer.size();
    m_bind_param_buffer[index] = std::move(buffer);

    MYSQL_BIND* param = &m_bind_param[index];
    param->buffer = m_bind_param_buffer[index].data(); 
    //param->buffer_length = m_bind_param_buffer[index].size();
    param->buffer_type = MYSQL_TYPE_DATETIME;
}

void PreparedStatement::setWholeNumber(const uint8 index, MYSQL_BIND* param, enum_field_types type,
    const void* src, uint32_t src_len, bool is_unsigned)
{
    m_bind_param_buffer[index].resize(src_len);
    std::memcpy(m_bind_param_buffer[index].data(), src, src_len);

    param->buffer_type = type;
    param->buffer = m_bind_param_buffer[index].data(); 
    param->is_unsigned = is_unsigned ? 1 : 0;
}

void PreparedStatement::setRealNumber(const uint8 index, MYSQL_BIND* param, enum_field_types type,
    const void* src, uint32_t src_len)
{
    m_bind_param_buffer[index].resize(src_len);
    std::memcpy(m_bind_param_buffer[index].data(), src, src_len);

    param->buffer_type = type;
    param->buffer = m_bind_param_buffer[index].data(); 
}

/*
std::string MySQLPreparedStatement::getQueryString(std::string const& sqlPattern) const
{
    std::string queryString = sqlPattern;

    size_t pos = 0;
    for (uint32 i = 0; i < m_stmt->statement_data.size(); i++)
    {
        pos = queryString.find('?', pos);
        std::stringstream ss;

        switch (m_stmt->statement_data[i].type)
        {
            case TYPE_BOOL:
                ss << uint16(m_stmt->statement_data[i].data.boolean);
                break;
            case TYPE_UI8:
                ss << uint16(m_stmt->statement_data[i].data.ui8);  // stringstream will append a character with that code instead of numeric representation
                break;
            case TYPE_UI16:
                ss << m_stmt->statement_data[i].data.ui16;
                break;
            case TYPE_UI32:
                ss << m_stmt->statement_data[i].data.ui32;
                break;
            case TYPE_I8:
                ss << int16(m_stmt->statement_data[i].data.i8);  // stringstream will append a character with that code instead of numeric representation
                break;
            case TYPE_I16:
                ss << m_stmt->statement_data[i].data.i16;
                break;
            case TYPE_I32:
                ss << m_stmt->statement_data[i].data.i32;
                break;
            case TYPE_UI64:
                ss << m_stmt->statement_data[i].data.ui64;
                break;
            case TYPE_I64:
                ss << m_stmt->statement_data[i].data.i64;
                break;
            case TYPE_FLOAT:
                ss << m_stmt->statement_data[i].data.f;
                break;
            case TYPE_DOUBLE:
                ss << m_stmt->statement_data[i].data.d;
                break;
            case TYPE_STRING:
                ss << '\'' << (char const*)m_stmt->statement_data[i].binary.data() << '\'';
                break;
            case TYPE_BINARY:
                ss << "BINARY";
                break;
            case TYPE_NULL:
                ss << "NULL";
                break;
        }

        std::string replaceStr = ss.str();
        queryString.replace(pos, 1, replaceStr);
        pos += replaceStr.length();
    }

    return queryString;
}
*/

}
