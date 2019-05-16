#include "Statement.h"

#include <errmsg.h>

#include <cstring>

#include "mysqlcpp/Connection.h"
#include "mysqlcpp/MysqlcppLog.h"
#include "mysqlcpp/Utility.h"
#include "mysqlcpp/ResultSet.h"
#include "mysqlcpp/Field.h"
#include "mysqlcpp/FieldMeta.h"

namespace mysqlcpp {

Statement::Statement(Connection& conn)
    : m_conn(conn)
{
}

Statement::~Statement()
{
}

Connection* Statement::getConnection()
{
    return &m_conn;
}

bool Statement::execute(const std::string& sql)
{
    if (!checkConnection())
        return false;
    auto* mysql = m_conn.getMYSQL();
    if (::mysql_real_query(mysql, sql.c_str(), static_cast<unsigned long>(sql.length()))) {
        uint32 err_no = ::mysql_errno(mysql);
        const char* err_str = ::mysql_error(mysql);
        MYSQLCPP_LOG(Error, "mysql_query %u:%s", err_no, err_str);
        m_conn.storeError(err_no, err_str);
        return false;
    }
    return true;
}

ResultSetPtr Statement::executeQuery(const std::string& sql)
{
    if (!checkConnection())
        return nullptr;
    MYSQL* mysql = m_conn.getMYSQL();
    if (::mysql_real_query(mysql, sql.c_str(), static_cast<unsigned long>(sql.length()))) {
        uint32 err_no = ::mysql_errno(mysql);
        const char* err_str = ::mysql_error(mysql);
        MYSQLCPP_LOG(Error, "mysql_query %u:%s", err_no, err_str);
        m_conn.storeError(err_no, err_str);
        return nullptr;
    }
    return createResultSet();
}

bool Statement::checkConnection()
{
    auto* mysql = m_conn.getMYSQL();
    if (!mysql) {
        m_conn.storeError(CR_UNKNOWN_ERROR, nullptr);
        return false;
    }
    return true;
}

ResultSetPtr Statement::createResultSet()
{
    /*
    * It is possible for mysql_store_result() to return NULL following a successful call to mysql_query().
    * When this happens, it means one of the following conditions occurred:
    *   There was a malloc() failure (for example, if the result set was too large).
    *   The data could not be read (an error occurred on the connection).
    *   The query returned no data (for example, it was an INSERT, UPDATE, or DELETE).
    * You can always check whether the statement should have produced a nonempty result by calling
    * mysql_field_count(). If mysql_field_count() returns zero, the result is empty and the
    * last query was a statement that does not return values (for example, an INSERT or a DELETE). If
    * mysql_field_count() returns a nonzero value, the statement should have produced a nonempty
    * result. See the description of the mysql_field_count() function for an example.
    * You can test for an error by calling mysql_error() or mysql_errno()
    */

    MYSQL* mysql = m_conn.getMYSQL();
    MYSQL_RES* mysql_res = ::mysql_store_result(mysql);
    if (!mysql_res) {
        unsigned int field_cnt = ::mysql_field_count(mysql);
        if (field_cnt > 0)
            return nullptr;
    }
    if (!mysql_res)
        return nullptr;
    FieldData fields_data{};
    utility::bindFiledsMeta(mysql_res, &fields_data);

    std::vector<RowData> rows{};
    if (!storeResult(mysql_res, &rows, &fields_data)) {
        ::mysql_free_result(mysql_res);
        return nullptr;
    }
    ::mysql_free_result(mysql_res);

    auto rs = std::make_shared<ResultSet>();
    rs->setRows(std::move(rows));
    rs->setFieldsMeta(std::move(fields_data));
    return rs;
}

bool Statement::storeResult(MYSQL_RES* mysql_res, std::vector<RowData>* all_row, const FieldData* field_data)
{
    /*
    *  mysql_num_rows() is intended for use with statements that return a result set, such as SELECT.For
    *  statements such as INSERT, UPDATE, or DELETE, the number of affected rows can be obtained with
    *  mysql_affected_rows().
    *
    * Because mysql_affected_rows() returns an unsigned value,
    * you can check for - 1 by comparing the return value to(my_ulonglong) - 1
    * (or to(my_ulonglong)~0, which is equivalent)
    */
    uint64_t row_count = ::mysql_num_rows(mysql_res);
    if (row_count == (my_ulonglong)~0)
        return false;

    std::vector<RowData> all_row_temp{};
    all_row_temp.reserve(row_count);

    MYSQL_ROW mysql_row{};
    while ((mysql_row = ::mysql_fetch_row(mysql_res))) {
        unsigned long* lengths = ::mysql_fetch_lengths(mysql_res);
        if (!lengths) {
            MYSQLCPP_LOG(Error, "mysql_fetch_lengths, cannot retrieve value lengths. Error %s",
                ::mysql_error(mysql_res->handle));
            return false;
        }

        RowData row{};
        row.resize(field_data->size());
        for (size_t i = 0; i != field_data->size(); ++i) {
            row[i].setBinaryValue((*field_data)[i].m_type, mysql_row[i], static_cast<uint32>(lengths[i]), false);
            if (mysql_row[i]) {
                row[i].setBinaryValue((*field_data)[i].m_type, mysql_row[i], static_cast<uint32>(lengths[i]), false);
            } else {
                row[i].setNullValue((*field_data)[i].m_type);
            }
        }
        all_row_temp.emplace_back(std::move(row));
    }
    if (all_row) {
        *all_row = std::move(all_row_temp);
    }
    return true;
}

} // mysqlcpp
