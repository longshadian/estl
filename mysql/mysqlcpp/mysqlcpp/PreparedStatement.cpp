#include "mysqlcpp/PreparedStatement.h"

#include <cstring>

#include "mysqlcpp/Connection.h"
#include "mysqlcpp/MysqlcppLog.h"
#include "mysqlcpp/Utility.h"
#include "mysqlcpp/Field.h"
#include "mysqlcpp/FieldMeta.h"
#include "mysqlcpp/ResultSet.h"
#include "mysqlcpp/Utility.h"

#include "mysqlcpp/detail/ParamBind.h"
#include "mysqlcpp/detail/ResultBind.h"

namespace mysqlcpp {

PreparedStatement::PreparedStatement(Connection& conn, MYSQL_STMT* stmt)
    : m_conn(conn)
    , m_stmt(stmt)
    , m_mysql_res()
    , m_param_count()
    , m_param_bind()
    , m_result_bind()
{
    /// "If set to 1, causes mysql_stmt_store_result() to update the metadata MYSQL_FIELD->max_length value."
    int temp = 1;
    ::mysql_stmt_attr_set(stmt, STMT_ATTR_UPDATE_MAX_LENGTH, &temp);

    m_param_count = ::mysql_stmt_param_count(m_stmt);
    m_param_bind = std::make_shared<detail::ParamBind>(m_param_count);
    m_result_bind = std::make_shared<detail::ResultBind>();
}

PreparedStatement::~PreparedStatement()
{
    if (m_mysql_res)
        ::mysql_free_result(m_mysql_res);
    if (m_stmt)
        ::mysql_stmt_close(m_stmt);
}

Connection* PreparedStatement::getConnection()
{
    return &m_conn;
}

bool PreparedStatement::execute()
{
    if (!bindParam())
        return false;
    if (!bindResult())
        return false;
    return true;
}

ResultSetPtr PreparedStatement::executeQuery()
{
    m_mysql_res = ::mysql_stmt_result_metadata(m_stmt);
    if (!m_mysql_res)
        return nullptr;

    std::vector<FieldMeta> fields_meta{};
    utility::bindFiledsMeta(m_mysql_res, &fields_meta);

    if (!bindParam())
        return nullptr;
    std::vector<std::vector<Field>> all_row{};
    if (!bindResult(&all_row, fields_meta.size()))
        return nullptr;
    auto rs = std::make_shared<ResultSet>();
    rs->setRows(std::move(all_row));
    rs->setFieldsMeta(std::move(fields_meta));
    return rs;
}

void PreparedStatement::clearParameters()
{
    m_param_bind->clear();
    m_result_bind->clear();
}

MYSQL_STMT* PreparedStatement::getMYSQL_STMT()
{ 
    return m_stmt; 
}

MYSQL_RES* PreparedStatement::getMySQL_RES()
{
    return m_mysql_res;
}

unsigned long long PreparedStatement::getAffectedRows()
{
    return ::mysql_stmt_affected_rows(m_stmt);
}

void PreparedStatement::setBool(uint32 index, bool value) { m_param_bind->setBool(index, value); }
void PreparedStatement::setUInt8(uint32 index, uint8 value) { m_param_bind->setUInt8(index, value); }
void PreparedStatement::setUInt16(uint32 index, uint16 value) { m_param_bind->setUInt16(index, value); }
void PreparedStatement::setUInt32(uint32 index, uint32 value) { m_param_bind->setUInt32(index, value); }
void PreparedStatement::setUInt64(uint32 index, uint64 value) { m_param_bind->setUInt64(index, value); }
void PreparedStatement::setInt8(uint32 index, int8 value) { m_param_bind->setInt8(index, value); }
void PreparedStatement::setInt16(uint32 index, int16 value) { m_param_bind->setInt16(index, value); }
void PreparedStatement::setInt32(uint32 index, int32 value) { m_param_bind->setInt32(index, value); }
void PreparedStatement::setInt64(uint32 index, int64 value) { m_param_bind->setInt64(index, value); }
void PreparedStatement::setFloat(uint32 index, float value) { m_param_bind->setFloat(index, value); }
void PreparedStatement::setDouble(uint32 index, double value) { m_param_bind->setDouble(index, value); }
void PreparedStatement::setString(uint32 index, const std::string& value) { m_param_bind->setString(index, value); }
void PreparedStatement::setString(uint32 index, const char* value) { m_param_bind->setString(index, value); }
void PreparedStatement::setBinary(uint32 index, std::vector<uint8> src, bool is_string) { m_param_bind->setBinary(index, src, is_string); }
void PreparedStatement::setNull(uint32 index) { m_param_bind->setNull(index); }
void PreparedStatement::setDateTime(uint32 index, const DateTime& tm) { m_param_bind->setDateTime(index, tm); }

bool PreparedStatement::bindParam()
{
    if (::mysql_stmt_bind_param(getMYSQL_STMT(), m_param_bind->getMYSQL_BIND())) {
        uint32 err_no = ::mysql_errno(m_conn.getMYSQL());
        const char* err_str = ::mysql_stmt_error(getMYSQL_STMT());
        MYSQLCPP_LOG(Error, "mysql_stmt_bind_param %u:%s", err_no, err_str);
        m_conn.storeError(err_no, err_str);
        return false;
    }

    if (::mysql_stmt_execute(getMYSQL_STMT())) {
        uint32 err_no = ::mysql_errno(m_conn.getMYSQL());
        const char* err_str = ::mysql_stmt_error(getMYSQL_STMT());
        MYSQLCPP_LOG(Error, "mysql_stmt_execute %d:%s", err_no, err_str);
        m_conn.storeError(err_no, err_str);
        return false;
    }
    return true;
}

bool PreparedStatement::bindResult(std::vector<RowData>* all_rows, size_t num_field)
{
    if (::mysql_stmt_store_result(getMYSQL_STMT())) {
        MYSQLCPP_LOG(Error, "mysql_stmt_store_result, cannot bind result from MySQL server. Error: %s", 
            ::mysql_stmt_error(getMYSQL_STMT()));
        return false;
    }

    if (!m_result_bind->bindResult(*this)) {
        ::mysql_stmt_free_result(getMYSQL_STMT());
        MYSQLCPP_LOG(Error, "result bind fail");
        return false;
    }

    unsigned long long num_rows = ::mysql_stmt_num_rows(getMYSQL_STMT());
    // û�з��ؽ��������insert update, delete
    if (num_rows == 0 || num_field == 0) {
        ::mysql_stmt_free_result(getMYSQL_STMT());
        return true;
    }

    unsigned long long row_pos = 0;
    std::vector<RowData> all_rows_temp{};
    all_rows_temp.reserve(num_rows);
    while (true) {
        auto ret = nextRow();
        if (ret == NEXT_ROW::FAILED) {
            ::mysql_stmt_free_result(getMYSQL_STMT());
            return false;
        } else if (ret == NEXT_ROW::NO_DATA_) {
            break;
        } else if (ret == NEXT_ROW::TRUNCATED) {
            // TODO
        }

        RowData row{};
        row.resize(num_field);
        for (size_t i = 0; i != num_field; ++i) {
            const MYSQL_BIND* rbind = m_result_bind->getMYSQL_BIND(i);
            if (*rbind->is_null) {
                row[i].setNullValue(rbind->buffer_type);
            } else {
                row[i].setBinaryValue(rbind->buffer_type, rbind->buffer, *rbind->length, true);
            }
        }
        all_rows_temp.emplace_back(std::move(row));
        ++row_pos;
    }
    //�᲻�����ʵ��������Ԥ����??
    if (row_pos < num_rows) {
        // TODO
    }
    ::mysql_stmt_free_result(getMYSQL_STMT());
    if (all_rows)
        *all_rows = std::move(all_rows_temp);
    return true;
}

PreparedStatement::NEXT_ROW PreparedStatement::nextRow()
{
    /*
    *  0 Successful, the data has been fetched to application data buffers.
    *  1 Error occurred.Error code and message can be obtained by calling
    *  mysql_stmt_errno() and mysql_stmt_error().
    *  MYSQL_NO_DATA No more rows / data exists
    *  MYSQL_DATA_TRUNCATED Data truncation occurred
    */
    int ret = ::mysql_stmt_fetch(getMYSQL_STMT());
    if (ret == 0) {
        return NEXT_ROW::SUCCESS;   // ������
    } else if (ret == 1) {
        return NEXT_ROW::FAILED;    // ����
    } else if (ret == MYSQL_NO_DATA) {
        return NEXT_ROW::NO_DATA_;   // û����
    } else if (ret == MYSQL_DATA_TRUNCATED) {
        return NEXT_ROW::TRUNCATED; // �ض�
    }
    return NEXT_ROW::FAILED;    // ����
}


}
