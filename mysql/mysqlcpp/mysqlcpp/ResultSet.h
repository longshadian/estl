#pragma once

#include <mysql.h>

#include <memory>
#include "mysqlcpp/Types.h"

namespace mysqlcpp {

class MYSQLCPP_EXPORT ResultRow
{
public:
    ResultRow(const RowData& row_data, const FieldData& field_data);
    ~ResultRow() = default;
    ResultRow(const ResultRow& rhs) = default;
    ResultRow& operator=(const ResultRow& rhs) = default;
public:
    const Field* operator[](std::string str) const;
    const Field* operator[](size_t index) const;
private:
    int findIndex(const std::string& str) const;
private:
    const RowData*      m_row_data;
    const FieldData*    m_field_data;
};

class MYSQLCPP_EXPORT ResultSet
{
    friend class Statement;
    friend class PreparedStatement;
public:
    ResultSet();
    ~ResultSet();

    ResultSet(ResultSet const& right) = delete;
    ResultSet& operator=(ResultSet const& right) = delete;
public:
    uint64 getRowCount() const { return m_rows.size(); }
    ResultRow getRow(uint64 index) const;

private:
    void setRows(std::vector<RowData> rows);
    void setFieldsMeta(FieldData fields_data);
private:
    std::vector<RowData> m_rows;
    FieldData            m_field_data;
};

}
