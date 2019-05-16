#include "mysqlcpp/FieldMeta.h"

#include <algorithm>
#include "mysqlcpp/MysqlcppLog.h"

namespace mysqlcpp {

static const char* fieldTypeName(enum_field_types type)
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

FieldMeta::FieldMeta(const MYSQL_FIELD* field, uint32 field_index)
    : m_table_name()
    , m_table_alias()
    , m_name()
    , m_alias()
    , m_type_name()
    , m_type(field->type)
    , m_index(field_index)
{
    if (field->org_table)
        m_table_name = field->org_table;
    if (field->table)
        m_table_alias = field->table;
    if (field->org_name)
        m_name = field->org_name;
    if (field->name)
        m_alias = field->name;
    m_type_name = fieldTypeName(m_type);

    /*
    FAKE_LOG(INFO) << "table_name:" << m_meta.m_table_name 
        << " " << "table_alias:" << m_meta.m_table_alias 
        << " " << "name:" << m_meta.m_name 
        << " " << "alias:" << m_meta.m_alias
        << " " << "type:" << m_meta.m_type 
        << " " << "index:" << m_meta.m_index;
        */
}

FieldMeta::FieldMeta(const FieldMeta& rhs)
    : m_table_name(rhs.m_table_name)
    , m_table_alias(rhs.m_table_alias)
    , m_name(rhs.m_name)
    , m_alias(rhs.m_alias)
    , m_type_name(rhs.m_type_name)
    , m_type(rhs.m_type)
    , m_index(rhs.m_index)
{
}

FieldMeta& FieldMeta::operator=(const FieldMeta& rhs)
{
    if (this != &rhs) {
        m_table_name = rhs.m_table_name;
        m_table_alias = rhs.m_table_alias;
        m_name = rhs.m_name;
        m_alias = rhs.m_alias;
        m_type_name = rhs.m_type_name;
        m_type = rhs.m_type;
        m_index = rhs.m_index;
    }
    return *this;
}

FieldMeta::FieldMeta(FieldMeta&& rhs)
    : m_table_name(std::move(rhs.m_table_name))
    , m_table_alias(std::move(rhs.m_table_alias))
    , m_name(std::move(rhs.m_name))
    , m_alias(std::move(rhs.m_alias))
    , m_type_name(std::move(rhs.m_type_name))
    , m_type(std::move(rhs.m_type))
    , m_index(std::move(rhs.m_index))
{
}

FieldMeta& FieldMeta::operator=(FieldMeta&& rhs)
{
    if (this != &rhs) {
        m_table_name    = std::move(rhs.m_table_name);
        m_table_alias   = std::move(rhs.m_table_alias);
        m_name          = std::move(rhs.m_name);
        m_alias         = std::move(rhs.m_alias);
        m_type_name     = std::move(rhs.m_type_name);
        m_type          = std::move(rhs.m_type);
        m_index         = std::move(rhs.m_index);
    }
    return *this;
}

} // mysqlcpp
