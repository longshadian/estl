#ifndef _MYSQLCPP_QUERYRESULT_H
#define _MYSQLCPP_QUERYRESULT_H

#include <memory>
#include <mysql.h>

#include "Field.h"
#include "Assert.h"

namespace mysqlcpp {

struct MetaData
{
    MetaData(MYSQL_FIELD* field, uint32 field_index);

    char const* m_table_name;
    char const* m_table_alias;
    char const* m_name;
    char const* m_alias;
    char const* m_type;
    uint32      m_index;
};

class ResultRow
{
public:
    ResultRow(const std::vector<Field>& fields, const std::vector<MetaData>& fields_meta);
    ~ResultRow() = default;
    ResultRow(const ResultRow& rhs) = default;
    ResultRow& operator=(const ResultRow& rhs) = default;
public:
    const Field* operator[](std::string str) const;
    const Field* operator[](size_t index) const;
private:
    int findIndex(const std::string& str) const;
private:
    const std::vector<Field>* m_fields;
    const std::vector<MetaData>* m_fields_meta;
};


class ResultSet
{
public:
    ResultSet(MYSQL& mysql);
    ~ResultSet();

    ResultSet(ResultSet const& right) = delete;
    ResultSet& operator=(ResultSet const& right) = delete;
public:
    bool init();

    uint64 getRowCount() const { return m_row_count; }
    uint32 getFieldCount() const { return m_field_count; }
    ResultRow getRow(uint32 index) const;
private:
    bool fetchRows();
private:
    MYSQL&              m_mysql;
    MYSQL_RES*          m_mysql_res;
    MYSQL_FIELD*        m_mysql_fields;
    uint64              m_row_count;
    uint32              m_field_count;
    std::vector<std::vector<Field>> m_rows;
    std::vector<MetaData>     m_fields_meta;
};


class PreparedResultSet
{
public:
    PreparedResultSet(MYSQL& mysql, MYSQL_STMT& stmt);
    ~PreparedResultSet();

    PreparedResultSet(PreparedResultSet const& right) = delete;
    PreparedResultSet& operator=(PreparedResultSet const& right) = delete;
public:
    bool init();

    uint64 getRowCount() const { return m_row_count; }
    uint32 getFieldCount() const { return m_field_count; }
    ResultRow getRow(uint32 index) const;
private:
    int nextRow();
    bool fetchRows();
private:
    MYSQL&          m_mysql;
    MYSQL_STMT&     m_mysql_stmt;
    MYSQL_RES*      m_mysql_res; 
    MYSQL_FIELD*    m_mysql_fields;

    uint64      m_row_count;
    uint64      m_row_position;
    uint32      m_field_count;

    std::vector<MYSQL_BIND>     m_out_bind;
    std::vector<my_bool>        m_out_is_null;
    std::vector<unsigned long>  m_out_length;
    std::vector<Field>          m_out_row;
    std::vector<std::vector<char>> m_out_bind_buffer;
    std::vector<std::vector<Field>> m_rows;
    std::vector<MetaData>     m_fields_meta;
};

}

#endif

