#include "mysqlcpp/Field.h"

#include <cstring>
#include <string_view>

#include "mysqlcpp/MysqlcppLog.h"
#include "mysqlcpp/DateTime.h"
#include "mysqlcpp/Utility.h"
#include "mysqlcpp/MysqlcppAssert.h"

#include "mysqlcpp/detail/Convert.h"

namespace mysqlcpp {

Field::Field()
    : m_type(MYSQL_TYPE_DECIMAL)
    , m_buffer()
    , m_is_binary()
    , m_is_null()
{
}

Field::~Field()
{
}

Field::Field(const Field& rhs)
    : m_type(rhs.m_type)
    , m_buffer(rhs.m_buffer)
    , m_is_binary(rhs.m_is_binary)
    , m_is_null(rhs.m_is_null)
{
}

Field& Field::operator=(const Field& rhs)
{
    if (this != &rhs) {
        m_type = rhs.m_type;
        m_buffer = rhs.m_buffer;
        m_is_binary = rhs.m_is_binary;
        m_is_null = rhs.m_is_null;
    }
    return *this;
}

Field::Field(Field&& rhs)
    : m_type(std::move(rhs.m_type))
    , m_buffer(std::move(rhs.m_buffer))
    , m_is_binary(std::move(rhs.m_is_binary))
    , m_is_null(std::move(rhs.m_is_null))
{
}

Field& Field::operator=(Field&& rhs)
{
    if (this != &rhs) {
        m_type = std::move(rhs.m_type);
        m_buffer = std::move(rhs.m_buffer);
        m_is_binary = std::move(rhs.m_is_binary);
        m_is_null = std::move(rhs.m_is_null);
    }
    return *this;
}

void Field::setBinaryValue(enum_field_types type, void* src, unsigned long src_len, bool raw_bytes)
{
    m_type = type;
    m_is_binary = raw_bytes;
    m_buffer.Clear();
    if (src) {
        m_buffer.AppendBuffer(src, src_len);
    }
}

void Field::setNullValue(enum_field_types type)
{
    m_type = type;
    m_is_null = true;
}

bool Field::getBool() const 
{ 
    return getUInt8() == 1; 
}

uint8 Field::getUInt8() const
{
    if (isNull())
        return 0;

    #ifdef TRINITY_DEBUG
    if (!isType(MYSQL_TYPE_TINY)) {
        MYSQLCPP_LOG(ERROR) << "Warning: GetUInt8() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
    #endif

    if (m_is_binary) {
        uint8 val = 0;
        std::memcpy(&val, m_buffer.GetPtr(), sizeof(val));
        return val;
    }
    return detail::Convert<uint8>::cvt_noexcept(m_buffer.AsStringView());
}

int8 Field::getInt8() const
{
    if (isNull())
        return 0;

    #ifdef TRINITY_DEBUG
    if (!isType(MYSQL_TYPE_TINY)) {
        MYSQLCPP_LOG(ERROR) << "Warning: GetInt8() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
    #endif

    if (m_is_binary) {
        int8 val = 0;
        std::memcpy(&val, m_buffer.GetPtr(), sizeof(val));
        return val;
    }
    return detail::Convert<int8>::cvt_noexcept(m_buffer.AsStringView());
}

uint16 Field::getUInt16() const
{
    if (isNull())
        return 0;

    #ifdef TRINITY_DEBUG
    if (!isType(MYSQL_TYPE_SHORT) && !isType(MYSQL_TYPE_YEAR)) {
        MYSQLCPP_LOG(ERROR) << "Warning: GetUInt16() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
    #endif

    if (m_is_binary) {
        uint16 val = 0;
        std::memcpy(&val, m_buffer.GetPtr(), sizeof(val));
        return val;
    }
    return detail::Convert<uint16>::cvt_noexcept(m_buffer.AsStringView());
}

int16 Field::getInt16() const
{
    if (isNull())
        return 0;

    #ifdef TRINITY_DEBUG
    if (!isType(MYSQL_TYPE_SHORT) && !isType(MYSQL_TYPE_YEAR)) {
        MYSQLCPP_LOG(ERROR) << "Warning: GetInt16() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
    #endif

    if (m_is_binary) {
        int16 val = 0;
        std::memcpy(&val, m_buffer.GetPtr(), sizeof(val));
        return val;
    }
    return detail::Convert<int16>::cvt_noexcept(m_buffer.AsStringView());
}

uint32 Field::getUInt32() const
{
    if (isNull())
        return 0;

    #ifdef TRINITY_DEBUG
    if (!isType(MYSQL_TYPE_INT24) && !isType(MYSQL_TYPE_LONG)) {
        MYSQLCPP_LOG(ERROR) << "Warning: GetUInt32() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
    #endif

    if (m_is_binary) {
        uint32 val = 0;
        std::memcpy(&val, m_buffer.GetPtr(), sizeof(val));
        return val;
    }
    return detail::Convert<uint32>::cvt_noexcept(m_buffer.AsStringView());
}

int32 Field::getInt32() const
{
    if (isNull())
        return 0;

    #ifdef TRINITY_DEBUG
    if (!isType(MYSQL_TYPE_INT24) && !isType(MYSQL_TYPE_LONG)) {
        MYSQLCPP_LOG(ERROR) << "Warning: GetInt32() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
    #endif

    if (m_is_binary) {
        int32 val = 0;
        std::memcpy(&val, m_buffer.GetPtr(), sizeof(val));
        return val;
    }
    return detail::Convert<int32>::cvt_noexcept(m_buffer.AsStringView());
}

uint64 Field::getUInt64() const
{
    if (isNull())
        return 0;

    #ifdef TRINITY_DEBUG
    if (!isType(MYSQL_TYPE_LONGLONG) && !isType(MYSQL_TYPE_BIT)) {
        MYSQLCPP_LOG(ERROR) << "Warning: GetUInt64() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
    #endif

    if (m_is_binary) {
        uint64 val = 0;
        std::memcpy(&val, m_buffer.GetPtr(), sizeof(val));
        return val;
    }
    return detail::Convert<uint64>::cvt_noexcept(m_buffer.AsStringView());
}

int64 Field::getInt64() const
{
    if (isNull())
        return 0;

    #ifdef TRINITY_DEBUG
    if (!isType(MYSQL_TYPE_LONGLONG) && !isType(MYSQL_TYPE_BIT)) {
        MYSQLCPP_LOG(ERROR) << "Warning: GetInt64() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
    #endif

    if (m_is_binary) {
        int64 val = 0;
        std::memcpy(&val, m_buffer.GetPtr(), sizeof(val));
        return val;
    }
    return detail::Convert<int64>::cvt_noexcept(m_buffer.AsStringView());
}

float Field::getFloat() const
{
    if (isNull())
        return 0.0f;

    #ifdef TRINITY_DEBUG
    if (!isType(MYSQL_TYPE_FLOAT)) {
        MYSQLCPP_LOG(ERROR) << "Warning: GetFloat() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
    #endif

    if (m_is_binary) {
        float val = 0;
        std::memcpy(&val, m_buffer.GetPtr(), sizeof(val));
        return val;
    }
    return detail::Convert<float>::cvt_noexcept(m_buffer.AsStringView());
}

double Field::getDouble() const
{
    if (isNull())
        return 0.0f;

    #ifdef TRINITY_DEBUG
    if (!isType(MYSQL_TYPE_DOUBLE) && !isType(MYSQL_TYPE_NEWDECIMAL)) {
        MYSQLCPP_LOG(ERROR) << "Warning: GetDouble() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
    #endif

    if (m_is_binary) {
        double val = 0;
        std::memcpy(&val, m_buffer.GetPtr(), sizeof(val));
        return val;
    }
    return detail::Convert<double>::cvt_noexcept(m_buffer.AsStringView());
}

long double Field::getLongDouble() const
{
    if (isNull())
        return 0.0f;
#ifdef TRINITY_DEBUG
    if (!isType(MYSQL_TYPE_DOUBLE) && !isType(MYSQL_TYPE_NEWDECIMAL)) {
        MYSQLCPP_LOG(ERROR) << "Warning: GetDouble() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
#endif

    if (m_is_binary) {
        long double val = 0;
        std::memcpy(&val, m_buffer.GetPtr(), sizeof(val));
        return val;
    }
    return detail::Convert<long double>::cvt_noexcept(m_buffer.AsStringView());
}

std::string_view Field::getStringView() const
{
    if (isNull())
        return std::string_view{};
    return m_buffer.AsStringView();
}

std::string Field::getString() const
{
    if (isNull())
        return "";
    return std::string(m_buffer.AsStringView());
}

std::vector<uint8> Field::getBinary() const
{
    if (isNull())
        return {};
    return m_buffer.AsBinary();
}

bool Field::isNull() const
{
    return m_is_null;
}

DateTime Field::getDateTime() const
{
    if (isNull())
        return DateTime{};

    MYSQL_TIME mysql_time{};
    utility::bzero(&mysql_time);
    if (m_is_binary){
        std::memcpy(&mysql_time, m_buffer.GetPtr(), m_buffer.Length());
        return DateTime(mysql_time);
    }
    if (m_type == MYSQL_TYPE_DATE) {
        utility::stringTo_Date(getString(), &mysql_time.year, &mysql_time.month, &mysql_time.day);
    }
    if (m_type == MYSQL_TYPE_DATETIME || m_type == MYSQL_TYPE_TIMESTAMP) {
        utility::stringTo_DateTime_Timestamp(getString()
            , &mysql_time.year, &mysql_time.month, &mysql_time.day
            , &mysql_time.hour, &mysql_time.minute, &mysql_time.second
            , &mysql_time.second_part);
    }
    return DateTime{mysql_time};
}

}
