#include "Field.h"

#include "FakeLog.h"
#include "DateTime.h"

namespace mysqlcpp {

Field::Slot::Slot()
    : m_type(MYSQL_TYPE_DECIMAL)
    , m_raw(false)
    , m_buffer()
    , m_length()
{
}

Field::Slot::~Slot()
{
}

Field::Slot::Slot(const Slot& rhs)
    : m_type(rhs.m_type)
    , m_raw(rhs.m_raw)
    , m_buffer(rhs.m_buffer)
    , m_length(rhs.m_length)
{
}

Field::Slot& Field::Slot::operator=(const Slot& rhs)
{
    if (this != &rhs) {
        m_type = rhs.m_type;
        m_raw = rhs.m_raw;
        m_buffer = rhs.m_buffer;
        m_length = rhs.m_length;
    }
    return *this;
}

Field::Slot::Slot(Slot&& rhs)
    : m_type(rhs.m_type)
    , m_raw(rhs.m_raw)
    , m_buffer(std::move(rhs.m_buffer))
    , m_length(rhs.m_length)
{
}

Field::Slot& Field::Slot::operator=(Slot&& rhs)
{
    if (this != &rhs) {
        m_type = rhs.m_type;
        m_raw = rhs.m_raw;
        m_buffer = std::move(rhs.m_buffer);
        m_length = rhs.m_length;
    }
    return *this;
}


Field::Field()
    : m_data()
{
}

Field::~Field()
{
}

Field::Field(const Field& rhs)
    : m_data(rhs.m_data)
{
}

Field& Field::operator=(const Field& rhs)
{
    if (this != &rhs) {
        m_data = rhs.m_data;
    }
    return *this;
}

Field::Field(Field&& rhs)
    : m_data(std::move(rhs.m_data))
{
}

Field& Field::operator=(Field&& rhs)
{
    if (this != &rhs) {
        m_data = std::move(rhs.m_data);
    }
    return *this;
}

uint32 Field::sizeForType(MYSQL_FIELD* field)
{
    switch (field->type)
    {
        case MYSQL_TYPE_NULL:
            return 0;
        case MYSQL_TYPE_TINY:
            return 1;
        case MYSQL_TYPE_YEAR:
        case MYSQL_TYPE_SHORT:
            return 2;
        case MYSQL_TYPE_INT24:
        case MYSQL_TYPE_LONG:
        case MYSQL_TYPE_FLOAT:
            return 4;
        case MYSQL_TYPE_DOUBLE:
        case MYSQL_TYPE_LONGLONG:
        case MYSQL_TYPE_BIT:
            return 8;
        case MYSQL_TYPE_TIMESTAMP:
        case MYSQL_TYPE_DATE:
        case MYSQL_TYPE_TIME:
        case MYSQL_TYPE_DATETIME:
            return sizeof(MYSQL_TIME);

        case MYSQL_TYPE_TINY_BLOB:
        case MYSQL_TYPE_MEDIUM_BLOB:
        case MYSQL_TYPE_LONG_BLOB:
        case MYSQL_TYPE_BLOB:
        case MYSQL_TYPE_STRING:
        case MYSQL_TYPE_VAR_STRING:
            return field->max_length;

        case MYSQL_TYPE_DECIMAL:
        case MYSQL_TYPE_NEWDECIMAL:
            return 64;

        case MYSQL_TYPE_GEOMETRY:
        /*
        Following types are not sent over the wire:
        MYSQL_TYPE_ENUM:
        MYSQL_TYPE_SET:
        */
        default:
            FAKE_LOG_ERROR() << "SQL::SizeForType(): invalid field type " << field->type;
            return 0;
    }
}


void Field::setByteValue(enum_field_types type, void* src, uint32 src_len, bool raw_bytes)
{
    m_data.m_type = type;
    m_data.m_raw = raw_bytes;
    if (src) {
        m_data.m_buffer.resize(src_len);
        m_data.m_length = src_len;
        std::memcpy(m_data.m_buffer.data(), src, src_len);
    } else {
        m_data.m_buffer.clear();
        m_data.m_length = 0;
    }
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
        FAKE_LOG_ERROR() << "Warning: GetUInt8() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
    #endif

    if (m_data.m_raw) {
        uint8 val = 0;
        std::memcpy(&val, m_data.m_buffer.data(), sizeof(val));
        return val;
    }
    return static_cast<uint8>(std::strtoul((char*)m_data.m_buffer.data(), nullptr, 10));
}

int8 Field::getInt8() const
{
    if (isNull())
        return 0;

    #ifdef TRINITY_DEBUG
    if (!isType(MYSQL_TYPE_TINY)) {
        FAKE_LOG_ERROR() << "Warning: GetInt8() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
    #endif

    if (m_data.m_raw) {
        int8 val = 0;
        std::memcpy(&val, m_data.m_buffer.data(), sizeof(val));
        return val;
    }
    return static_cast<int8>(std::strtol((char*)m_data.m_buffer.data(), NULL, 10));
}

uint16 Field::getUInt16() const
{
    if (isNull())
        return 0;

    #ifdef TRINITY_DEBUG
    if (!isType(MYSQL_TYPE_SHORT) && !isType(MYSQL_TYPE_YEAR)) {
        FAKE_LOG_ERROR() << "Warning: GetUInt16() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
    #endif

    if (m_data.m_raw) {
        uint16 val = 0;
        std::memcpy(&val, m_data.m_buffer.data(), sizeof(val));
        return val;
    }
    return static_cast<uint16>(std::strtoul((char*)m_data.m_buffer.data(), nullptr, 10));
}

int16 Field::getInt16() const
{
    if (isNull())
        return 0;

    #ifdef TRINITY_DEBUG
    if (!isType(MYSQL_TYPE_SHORT) && !isType(MYSQL_TYPE_YEAR)) {
        FAKE_LOG_ERROR() << "Warning: GetInt16() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
    #endif

    if (m_data.m_raw) {
        int16 val = 0;
        std::memcpy(&val, m_data.m_buffer.data(), sizeof(val));
        return val;
    }
    return static_cast<int16>(std::strtol((char*)m_data.m_buffer.data(), NULL, 10));
}

uint32 Field::getUInt32() const
{
    if (isNull())
        return 0;

    #ifdef TRINITY_DEBUG
    if (!isType(MYSQL_TYPE_INT24) && !isType(MYSQL_TYPE_LONG)) {
        FAKE_LOG_ERROR() << "Warning: GetUInt32() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
    #endif

    if (m_data.m_raw) {
        uint32 val = 0;
        std::memcpy(&val, m_data.m_buffer.data(), sizeof(val));
        return val;
    }
    return static_cast<uint32>(std::strtoul((char*)m_data.m_buffer.data(), nullptr, 10));
}

int32 Field::getInt32() const
{
    if (isNull())
        return 0;

    #ifdef TRINITY_DEBUG
    if (!isType(MYSQL_TYPE_INT24) && !isType(MYSQL_TYPE_LONG)) {
        FAKE_LOG_ERROR() << "Warning: GetInt32() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
    #endif

    if (m_data.m_raw) {
        int32 val = 0;
        std::memcpy(&val, m_data.m_buffer.data(), sizeof(val));
        return val;
    }
    return static_cast<int32>(std::strtol((char*)m_data.m_buffer.data(), NULL, 10));
}

uint64 Field::getUInt64() const
{
    if (isNull())
        return 0;

    #ifdef TRINITY_DEBUG
    if (!isType(MYSQL_TYPE_LONGLONG) && !isType(MYSQL_TYPE_BIT)) {
        FAKE_LOG_ERROR() << "Warning: GetUInt64() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
    #endif

    if (m_data.m_raw) {
        uint64 val = 0;
        std::memcpy(&val, m_data.m_buffer.data(), sizeof(val));
        return val;
    }
    return static_cast<uint64>(std::strtoull((char*)m_data.m_buffer.data(), nullptr, 10));
}

int64 Field::getInt64() const
{
    if (isNull())
        return 0;

    #ifdef TRINITY_DEBUG
    if (!isType(MYSQL_TYPE_LONGLONG) && !isType(MYSQL_TYPE_BIT)) {
        FAKE_LOG_ERROR() << "Warning: GetInt64() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
    #endif

    if (m_data.m_raw) {
        int64 val = 0;
        std::memcpy(&val, m_data.m_buffer.data(), sizeof(val));
        return val;
    }
    return static_cast<int64>(std::strtoll((char*)m_data.m_buffer.data(), NULL, 10));
}

float Field::getFloat() const
{
    if (isNull())
        return 0.0f;

    #ifdef TRINITY_DEBUG
    if (!isType(MYSQL_TYPE_FLOAT)) {
        FAKE_LOG_ERROR() << "Warning: GetFloat() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
    #endif

    if (m_data.m_raw) {
        float val = 0;
        std::memcpy(&val, m_data.m_buffer.data(), sizeof(val));
        return val;
    }
    return static_cast<float>(std::strtof((char*)m_data.m_buffer.data(), nullptr));
}

double Field::getDouble() const
{
    if (isNull())
        return 0.0f;

    #ifdef TRINITY_DEBUG
    if (!isType(MYSQL_TYPE_DOUBLE) && !isType(MYSQL_TYPE_NEWDECIMAL)) {
        FAKE_LOG_ERROR() << "Warning: GetDouble() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
    #endif

    if (m_data.m_raw) {
        double val = 0;
        std::memcpy(&val, m_data.m_buffer.data(), sizeof(val));
        return val;
    }
    return std::strtod((char*)m_data.m_buffer.data(), nullptr);
}

long double Field::getLongDouble() const
{
    if (isNull())
        return 0.0f;
#ifdef TRINITY_DEBUG
    if (!isType(MYSQL_TYPE_DOUBLE) && !isType(MYSQL_TYPE_NEWDECIMAL)) {
        FAKE_LOG_ERROR() << "Warning: GetDouble() on non-tinyint field " << m_meta.m_table_alias << " " << m_meta.m_alias << " " << m_meta.m_table_name << " " << m_meta.m_name << " " << m_meta.m_index << " " << m_meta.m_type;
        return 0;
    }
#endif

    if (m_data.m_raw) {
        long double val = 0;
        std::memcpy(&val, m_data.m_buffer.data(), sizeof(val));
        return val;
    }
    return std::strtold((char*)m_data.m_buffer.data(), nullptr);
}

/*
char const* Field::getCString() const
{
    if (!data.value)
        return NULL;

    #ifdef TRINITY_DEBUG
    if (isNumeric())
    {
        FAKE_LOG_ERROR() << "Warning: GetCString() on non-tinyint field " << meta.TableAlias
            << " " << meta.Alias
            << " " << meta.TableName
            << " " << meta.Name
            << " " << meta.Index
            << " " << meta.Type;
        return NULL;
    }
    #endif
    return static_cast<char const*>(data.value);
}
*/

std::string Field::getString() const
{
    if (isNull())
        return "";
    const char* p = (const char*)m_data.m_buffer.data();
    return std::string(p, p + m_data.m_length);
}

std::vector<uint8> Field::getBinary() const
{
    if (isNull())
        return {};
    return std::vector<uint8>(m_data.m_buffer.data(), m_data.m_buffer.data() + m_data.m_length);
}

bool Field::isNull() const
{
    return m_data.m_length == 0;
}

DateTime Field::getDateTime() const
{
    return DateTime(getString(), m_data.m_type);
}

bool Field::isType(enum_field_types type) const
{
    return m_data.m_type == type;
}

bool Field::isNumeric() const
{
    return (m_data.m_type == MYSQL_TYPE_TINY ||
            m_data.m_type == MYSQL_TYPE_SHORT ||
            m_data.m_type == MYSQL_TYPE_INT24 ||
            m_data.m_type == MYSQL_TYPE_LONG ||
            m_data.m_type == MYSQL_TYPE_FLOAT ||
            m_data.m_type == MYSQL_TYPE_DOUBLE ||
            m_data.m_type == MYSQL_TYPE_LONGLONG );
}

char const* Field::fieldTypeToString(enum_field_types type)
{
    switch (type)
    {
        case MYSQL_TYPE_BIT:         return "BIT";
        case MYSQL_TYPE_BLOB:        return "BLOB";
        case MYSQL_TYPE_DATE:        return "DATE";
        case MYSQL_TYPE_DATETIME:    return "DATETIME";
        case MYSQL_TYPE_NEWDECIMAL:  return "NEWDECIMAL";
        case MYSQL_TYPE_DECIMAL:     return "DECIMAL";
        case MYSQL_TYPE_DOUBLE:      return "DOUBLE";
        case MYSQL_TYPE_ENUM:        return "ENUM";
        case MYSQL_TYPE_FLOAT:       return "FLOAT";
        case MYSQL_TYPE_GEOMETRY:    return "GEOMETRY";
        case MYSQL_TYPE_INT24:       return "INT24";
        case MYSQL_TYPE_LONG:        return "LONG";
        case MYSQL_TYPE_LONGLONG:    return "LONGLONG";
        case MYSQL_TYPE_LONG_BLOB:   return "LONG_BLOB";
        case MYSQL_TYPE_MEDIUM_BLOB: return "MEDIUM_BLOB";
        case MYSQL_TYPE_NEWDATE:     return "NEWDATE";
        case MYSQL_TYPE_NULL:        return "NULL";
        case MYSQL_TYPE_SET:         return "SET";
        case MYSQL_TYPE_SHORT:       return "SHORT";
        case MYSQL_TYPE_STRING:      return "STRING";
        case MYSQL_TYPE_TIME:        return "TIME";
        case MYSQL_TYPE_TIMESTAMP:   return "TIMESTAMP";
        case MYSQL_TYPE_TINY:        return "TINY";
        case MYSQL_TYPE_TINY_BLOB:   return "TINY_BLOB";
        case MYSQL_TYPE_VAR_STRING:  return "VAR_STRING";
        case MYSQL_TYPE_YEAR:        return "YEAR";
        default:                     return "-Unknown-";
    }
}

}
