#include "mysqlcpp/detail/ResultBind.h"

#include <cstring>

#include "mysqlcpp/PreparedStatement.h"
#include "mysqlcpp/MysqlcppLog.h"

namespace mysqlcpp {

namespace detail {

static 
std::shared_ptr<ResultBindBuffer> createBindBufferForField(const MYSQL_FIELD* const field)
{
  switch (field->type)
  {
    case MYSQL_TYPE_NULL:
      return std::make_shared<ResultBindBuffer>(0, field->type);
    case MYSQL_TYPE_TINY:
      return std::make_shared<ResultBindBuffer>(1, field->type);
    case MYSQL_TYPE_SHORT:
      return std::make_shared<ResultBindBuffer>(2, field->type);
    case MYSQL_TYPE_INT24:
    case MYSQL_TYPE_LONG:
    case MYSQL_TYPE_FLOAT:
      return std::make_shared<ResultBindBuffer>(4, field->type);
    case MYSQL_TYPE_DOUBLE:
    case MYSQL_TYPE_LONGLONG:
      return std::make_shared<ResultBindBuffer>(8, field->type);
    case MYSQL_TYPE_YEAR:
      return std::make_shared<ResultBindBuffer>(2, MYSQL_TYPE_SHORT);
    case MYSQL_TYPE_TIMESTAMP:
    case MYSQL_TYPE_DATE:
    case MYSQL_TYPE_TIME:
    case MYSQL_TYPE_DATETIME:
      return std::make_shared<ResultBindBuffer>(sizeof(MYSQL_TIME), field->type);


    case MYSQL_TYPE_TINY_BLOB:
    case MYSQL_TYPE_MEDIUM_BLOB:
    case MYSQL_TYPE_LONG_BLOB:
    case MYSQL_TYPE_BLOB:
    case MYSQL_TYPE_STRING:
    case MYSQL_TYPE_VAR_STRING:
#if LIBMYSQL_VERSION_ID > 50700
    case MYSQL_TYPE_JSON:
      return std::make_shared<ResultBindBuffer>(field->max_length + 1, field->type);
#endif //LIBMYSQL_VERSION_ID > 50700

    case MYSQL_TYPE_DECIMAL:
    case MYSQL_TYPE_NEWDECIMAL:
      return std::make_shared<ResultBindBuffer>(64, field->type);
#if A1
    case MYSQL_TYPE_TIMESTAMP:
    case MYSQL_TYPE_YEAR:
      return std::make_shared<ResultBindBuffer>(10, field->type);
#endif
#if A0
      // There two are not sent over the wire
    case MYSQL_TYPE_ENUM:
    case MYSQL_TYPE_SET:
#endif
    case MYSQL_TYPE_BIT:
      return std::make_shared<ResultBindBuffer>(8, MYSQL_TYPE_BIT);
    case MYSQL_TYPE_GEOMETRY:
    default:
        return nullptr;
  }
}

ResultBind::ResultBind()
    : m_bind()
    , m_is_null()
    , m_err()
    , m_len()
    , m_bind_buffer()
{
}

ResultBind::~ResultBind()
{
}

bool ResultBind::bindResult(PreparedStatement& ps)
{
    clear();
    MYSQL_RES* mysql_res = ps.getMySQL_RES();
    if (!mysql_res)
        return true;
    auto num_fields = ::mysql_num_fields(mysql_res);
    if (num_fields == 0) {
        return true;
    }

    m_bind.resize(num_fields);
    std::memset(m_bind.data(), 0, sizeof(MYSQL_BIND) * num_fields);

    m_is_null.resize(num_fields, {false});
    //std::memset(m_is_null.data(), 0, sizeof(bool) * num_fields);

    m_err.resize(num_fields, { false });
    //std::memset(m_err.data(), 0, sizeof(bool) * num_fields);

    m_len.resize(num_fields, 0);
    //std::memset(m_len.data(), 0, sizeof(unsigned long) * num_fields);

    MYSQL_FIELD* mysql_fields = ::mysql_fetch_fields(mysql_res);
    for (size_t i = 0; i < num_fields; ++i) {
        MYSQL_FIELD * field = mysql_fields + i;
        auto p = createBindBufferForField(field);
        m_bind_buffer.push_back(p);
        m_bind[i].buffer_type= p->m_type;
        m_bind[i].buffer		= p->m_buffer;
        m_bind[i].buffer_length= static_cast<unsigned long>(p->m_size);
        m_bind[i].length		= &m_len[i];
        m_bind[i].is_null	= &m_is_null[i].m_bool;
        m_bind[i].error		= &m_err[i].m_bool;
        m_bind[i].is_unsigned= field->flags & UNSIGNED_FLAG;
    }

    if (::mysql_stmt_bind_result(ps.getMYSQL_STMT(), m_bind.data())) {
        MYSQLCPP_LOG(Error, "mysql_stmt_bind_result, cannot bind result from MySQL server. Error: ",
            ::mysql_stmt_error(ps.getMYSQL_STMT()));
        return false;
    }
    return true;
}

const MYSQL_BIND* ResultBind::getMYSQL_BIND(size_t i) const
{
    return &m_bind[i];
}

void ResultBind::clear()
{
    m_bind.clear();
    m_is_null.clear();
    m_err.clear();
    m_len.clear();
    m_bind_buffer.clear();
}

} // detail

} // mysqlcpp 
