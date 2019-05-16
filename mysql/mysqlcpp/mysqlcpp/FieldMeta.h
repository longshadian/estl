#pragma once

#include <memory>
#include <mysql.h>

#include "mysqlcpp/Field.h"

namespace mysqlcpp {

class MYSQLCPP_EXPORT FieldMeta
{
public:
    FieldMeta(const MYSQL_FIELD* field, uint32 field_index);
    ~FieldMeta() = default;

    FieldMeta(const FieldMeta& rhs);
    FieldMeta& operator=(const FieldMeta& rhs);

    FieldMeta(FieldMeta&& rhs);
    FieldMeta& operator=(FieldMeta&& rhs);

    std::string m_table_name;
    std::string m_table_alias;
    std::string m_name;
    std::string m_alias;
    std::string m_type_name;
    enum_field_types m_type;
    uint32      m_index;
};

} // mysqlcpp
