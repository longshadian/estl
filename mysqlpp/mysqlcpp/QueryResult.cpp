#include "QueryResult.h"

#include <algorithm>
#include "FakeLog.h"

namespace mysqlcpp {

MetaData::MetaData(MYSQL_FIELD* field, uint32 field_index)
{
    m_table_name = field->org_table;
    m_table_alias = field->table;
    m_name = field->org_name;
    m_alias = field->name;
    m_type = Field::fieldTypeToString(field->type);
    m_index = field_index;

    /*
    FAKE_LOG_INFO() << "table_name:" << m_meta.m_table_name 
        << " " << "table_alias:" << m_meta.m_table_alias 
        << " " << "name:" << m_meta.m_name 
        << " " << "alias:" << m_meta.m_alias
        << " " << "type:" << m_meta.m_type 
        << " " << "index:" << m_meta.m_index;
        */
}

ResultRow::ResultRow(const std::vector<Field>& fields, const std::vector<MetaData>& fields_meta)
    : m_fields(&fields)
    , m_fields_meta(&fields_meta)
{
}

const Field* ResultRow::operator[](std::string str) const
{
    auto idx = findIndex(str);
    if (idx < 0)
        return nullptr;
    return &(*m_fields)[idx];
}

const Field* ResultRow::operator[](size_t index) const
{
    ASSERT(index < m_fields->size());
    return &(*m_fields)[index];
}

int ResultRow::findIndex(const std::string& str) const
{
    auto it = std::find_if(m_fields_meta->begin(), m_fields_meta->end(),
        [&str](const MetaData& meta) { return meta.m_name == str || meta.m_alias == str; }); 
    if (it == m_fields_meta->end())
        return -1;
    return (int)it->m_index;
}


ResultSet::ResultSet(MYSQL& mysql)
    : m_mysql(mysql)
    , m_mysql_res(nullptr)
    , m_mysql_fields(nullptr)
    , m_row_count()
    , m_field_count()
    , m_rows()
    , m_fields_meta()
{
}

ResultSet::~ResultSet()
{
    if (m_mysql_res)
        ::mysql_free_result(m_mysql_res);
}

bool ResultSet::init()
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
    m_mysql_res = ::mysql_store_result(&m_mysql);
    if (!m_mysql_res) {
        unsigned int field_cnt = ::mysql_field_count(&m_mysql);
        if (field_cnt > 0)
            return false;
    }
    if (!m_mysql_res)
        return true;

    /*
     *  mysql_num_rows() is intended for use with statements that return a result set, such as SELECT.For
     *  statements such as INSERT, UPDATE, or DELETE, the number of affected rows can be obtained with
     *  mysql_affected_rows().
     *
     * Because mysql_affected_rows() returns an unsigned value,
     * you can check for - 1 by comparing the return value to(my_ulonglong) - 1 
     * (or to(my_ulonglong)~0, which is equivalent)
     */
    m_row_count = ::mysql_num_rows(m_mysql_res);
    if (m_row_count == (my_ulonglong)~0)
        return false;

    m_mysql_fields = ::mysql_fetch_fields(m_mysql_res);
    m_field_count = ::mysql_num_fields(m_mysql_res);
    if (!fetchRows()) {
        return false;
    }
    return true;
}

ResultRow ResultSet::getRow(uint32 index) const
{
    ASSERT(index < m_row_count);
    return ResultRow(m_rows[index], m_fields_meta);
}

bool ResultSet::fetchRows()
{
    //没有数据产生
    if (!m_mysql_res)
        return true;

    //获取列元数据
    m_fields_meta.reserve(m_field_count);
    for (uint32 i = 0; i != m_field_count; ++i) {
        m_fields_meta.push_back(MetaData(&m_mysql_fields[i], i));
    }

    m_rows.resize(m_row_count);
    for (uint32 i = 0; i != m_row_count; ++i) {
        MYSQL_ROW row = ::mysql_fetch_row(m_mysql_res);
        if (!row) {
            return false;
        }
        unsigned long* lengths = ::mysql_fetch_lengths(m_mysql_res);
        if (!lengths) {
            FAKE_LOG_ERROR() << "mysql_fetch_lengths, cannot retrieve value lengths. Error " << ::mysql_error(m_mysql_res->handle);
            return false;
        }
        std::vector<Field> fields{};
        fields.resize(m_field_count);
        for (uint32 j = 0; j < m_field_count; ++j) {
            fields[j].setByteValue(m_mysql_fields[j].type, row[j], lengths[j], false);
        }
        m_rows[i] = std::move(fields);
    }
    return true;
}


PreparedResultSet::PreparedResultSet(MYSQL& mysql, MYSQL_STMT& stmt) 
    : m_mysql(mysql)
    , m_mysql_stmt(stmt)
    , m_mysql_res(nullptr)
    , m_row_count()
    , m_row_position()
    , m_field_count()
    , m_out_bind()
    , m_out_is_null()
    , m_out_length()
    , m_out_row()
    , m_out_bind_buffer()
    , m_rows()
{
}

PreparedResultSet::~PreparedResultSet()
{
    if (m_mysql_res)
        ::mysql_free_result(m_mysql_res);
}

bool PreparedResultSet::init()
{
    m_mysql_res = ::mysql_stmt_result_metadata(&m_mysql_stmt);
    if (!m_mysql_res)
        return true;

    /*
     *  mysql_num_rows() is intended for use with statements that return a result set, such as SELECT.For
     *  statements such as INSERT, UPDATE, or DELETE, the number of affected rows can be obtained with
     *  mysql_affected_rows().
     *
     * Because mysql_affected_rows() returns an unsigned value,
     * you can check for - 1 by comparing the return value to(my_ulonglong) - 1 
     * (or to(my_ulonglong)~0, which is equivalent)
     */
    m_row_count = ::mysql_num_rows(m_mysql_res);
    if (m_row_count == (my_ulonglong)~0)
        return false;

    m_mysql_fields = ::mysql_fetch_fields(m_mysql_res);
    m_field_count = ::mysql_num_fields(m_mysql_res);
    if (!fetchRows()) {
        return false;
    }
    return true;
}

int PreparedResultSet::nextRow()
{
    if (m_row_position >= m_row_count)
        return 0;   //没数据

    /*
     *  0 Successful, the data has been fetched to application data buffers.
     *  1 Error occurred.Error code and message can be obtained by calling
     *  mysql_stmt_errno() and mysql_stmt_error().
     *  MYSQL_NO_DATA No more rows / data exists
     *  MYSQL_DATA_TRUNCATED Data truncation occurred
     */
    int val = ::mysql_stmt_fetch(&m_mysql_stmt);
    if (val == 0)
        return 1;  //有数据
    if (val == 1)
        return -1; //出错
    if (val == MYSQL_NO_DATA)
        return 0;  //没数据
    if (val == MYSQL_DATA_TRUNCATED)
        return -1;  //出错
    return 0;
}

bool PreparedResultSet::fetchRows()
{
    if (!m_mysql_res)
        return true;
    m_field_count = ::mysql_num_fields(m_mysql_res);

    if (::mysql_stmt_store_result(&m_mysql_stmt)) {
        FAKE_LOG_ERROR() << "mysql_stmt_store_result, cannot bind result from MySQL server. Error:" << ::mysql_stmt_error(&m_mysql_stmt);
        return false;
    }

    m_out_bind.resize(m_field_count);
    m_out_is_null.resize(m_field_count);
    m_out_length.resize(m_field_count);
    m_out_bind_buffer.resize(m_field_count);

    std::memset(m_out_bind.data(), 0, sizeof(MYSQL_BIND) * m_out_bind.size());
    std::memset(m_out_is_null.data(), 0, sizeof(my_bool) * m_out_is_null.size());
    std::memset(m_out_length.data(), 0, sizeof(unsigned long) * m_out_length.size());

    m_row_count = ::mysql_stmt_num_rows(&m_mysql_stmt);

    m_fields_meta.reserve(m_field_count);
    for (uint32 i = 0; i != m_field_count; ++i) {
        //保存列元数据
        m_fields_meta.push_back(MetaData(&m_mysql_fields[i], i));

        // 准备buffer
        uint32 size = Field::sizeForType(&m_mysql_fields[i]);
        std::vector<char> buffer{};
        buffer.resize(size);
        m_out_bind_buffer[i] = std::move(buffer);
        m_out_bind[i].buffer = m_out_bind_buffer[i].data();
        m_out_bind[i].buffer_length = size;
        m_out_bind[i].buffer_type = m_mysql_fields[i].type;
        m_out_bind[i].length = &m_out_length[i];
        m_out_bind[i].is_null = &m_out_is_null[i];
        m_out_bind[i].error = nullptr;
        m_out_bind[i].is_unsigned = m_mysql_fields[i].flags & UNSIGNED_FLAG;
    }

    if (::mysql_stmt_bind_result(&m_mysql_stmt, m_out_bind.data())) {
        FAKE_LOG_ERROR() << "mysql_stmt_bind_result, cannot bind result from MySQL server. Error: " << ::mysql_stmt_error(&m_mysql_stmt);
        ::mysql_stmt_free_result(&m_mysql_stmt);
        return false;
    }

    m_rows.resize(m_row_count);
    size_t m_row_position = 0;
    while (true) {
        int ret = nextRow();
        if (ret == -1)
            return false;
        else if (ret == 0)
            break;

        std::vector<Field> fields{};
        fields.resize(m_field_count);
        for (uint32 i = 0; i < m_field_count; ++i) {
            unsigned long fetched_length = *m_out_bind[i].length;
            if (!*m_out_bind[i].is_null) {
                void* buffer = m_out_bind[i].buffer;
                fields[i].setByteValue(m_out_bind[i].buffer_type, buffer, fetched_length, true);
            } else {
                fields[i].setByteValue(m_out_bind[i].buffer_type, nullptr, *m_out_bind[i].length, true);
            }
        }
        m_rows[m_row_position] = std::move(fields);
        m_row_position++;
    }
    //会不会存在实际行数比预定少??
    if (m_row_position < m_row_count) {
        m_row_count= m_row_position;
        m_rows.resize(m_row_count);
    }

    /// All data is buffered, let go of mysql c api structures
    ::mysql_stmt_free_result(&m_mysql_stmt);
    return true;
}

ResultRow PreparedResultSet::getRow(uint32 index) const
{
    ASSERT(index < m_row_count);
    return ResultRow(m_rows[index], m_fields_meta);
}

}
