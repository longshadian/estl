#include "mysqlcpp/ResultSet.h"

#include <algorithm>
#include "mysqlcpp/MysqlcppAssert.h"
#include "mysqlcpp/MysqlcppLog.h"
#include "mysqlcpp/Field.h"
#include "mysqlcpp/FieldMeta.h"

namespace mysqlcpp {

ResultRow::ResultRow(const RowData& row_data, const FieldData& field_data)
    : m_row_data(&row_data)
    , m_field_data(&field_data)
{
}

const Field* ResultRow::operator[](std::string str) const
{
    auto idx = findIndex(str);
    if (idx < 0)
        return nullptr;
    return &(*m_row_data)[idx];
}

const Field* ResultRow::operator[](size_t index) const
{
    MYSQLCPP_ASSERT(index < m_row_data->size());
    return &(*m_row_data)[index];
}

int ResultRow::findIndex(const std::string& str) const
{
    auto it = std::find_if(m_field_data->begin(), m_field_data->end(),
        [&str](const FieldMeta& meta) { return meta.m_name == str || meta.m_alias == str; });
    if (it == m_field_data->end())
        return -1;
    return (int)it->m_index;
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
ResultSet::ResultSet()
    : m_rows()
    , m_field_data()
{
}

ResultSet::~ResultSet()
{
}

void ResultSet::setRows(std::vector<RowData> rows)
{
    m_rows = std::move(rows);
}

void ResultSet::setFieldsMeta(FieldData fields_data)
{
    m_field_data = std::move(fields_data);
}

ResultRow ResultSet::getRow(uint64 index) const
{
    MYSQLCPP_ASSERT(index < getRowCount());
    return ResultRow(m_rows[static_cast<uint32_t>(index)], m_field_data);
}

} // mysqlcpp
