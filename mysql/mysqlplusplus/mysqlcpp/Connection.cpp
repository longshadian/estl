#include "Connection.h"

#include <mysql.h>
#include <errmsg.h>

#include "QueryResult.h"
#include "PreparedStatement.h"
#include "FakeLog.h"
#include "Utils.h"

namespace mysqlcpp {

Connection::Connection(ConnectionOpt conn_opt) 
    : m_reconnecting(false)
    , m_mysql(nullptr)
    , m_conn_info(conn_opt)
{
}

Connection::~Connection()
{
    close();
}

void Connection::close()
{
    if (m_mysql) {
        ::mysql_close(m_mysql);
        m_mysql = nullptr;
    }
}

PreparedResultSetPtr Connection::query(PreparedStatement& stmt)
{
    if (!queryDetail(stmt))
        return nullptr;
    if (::mysql_more_results(m_mysql)) {
        ::mysql_next_result(m_mysql);
    }
    auto result = std::make_shared<PreparedResultSet>(*m_mysql, *stmt.getMYSQL_STMT());
    if (!result->init()) {
        return nullptr;
    }
    return result;
}

ResultSetPtr Connection::query(const char* sql)
{
    if (!sql)
        return nullptr;
    if (!queryDetail(sql))
        return nullptr;

    auto result = std::make_shared<ResultSet>(*m_mysql);
    if (!result->init()) {
        return nullptr;
    }
    return result;
}

uint32 Connection::open()
{
    MYSQL* mysql;
    mysql = ::mysql_init(nullptr);
    if (!mysql) {
        FAKE_LOG_ERROR() << "Could not initialize Mysql connection to database " << m_conn_info.database;
        return CR_UNKNOWN_ERROR;
    }

    ::mysql_options(mysql, MYSQL_SET_CHARSET_NAME, "utf8");

    //unsigned int timeout = 10;
    //::mysql_options(mysqlInit, MYSQL_OPT_READ_TIMEOUT, (char const*)&timeout);
    const char* host = m_conn_info.host.c_str();
    const char* user = m_conn_info.user.c_str();
    const char* passwd = m_conn_info.password.c_str();
    const char* db = m_conn_info.database.c_str();
    db = nullptr;

    m_mysql = ::mysql_real_connect(mysql, host, user, passwd, db, m_conn_info.port, nullptr, 0);

    if (m_mysql) {
        if (!m_reconnecting) {
            FAKE_LOG_INFO() << "MySQL client library:" << ::mysql_get_client_info();
            FAKE_LOG_INFO() << "MySQL server ver: " << ::mysql_get_server_info(m_mysql);
        }

        FAKE_LOG_INFO() << "Connected to MySQL database at " << m_conn_info.host;
        ::mysql_autocommit(m_mysql, 1);

        // set connection properties to UTF8 to properly handle locales for different
        // server configs - core sends data in UTF8, so MySQL must expect UTF8 too
        ::mysql_set_character_set(m_mysql, "utf8");
        return 0;
    } else {
        FAKE_LOG_ERROR() << "Could not connect to MySQL database at " << m_conn_info.host.c_str() << " : " << ::mysql_error(mysql);
        ::mysql_close(mysql);
        return ::mysql_errno(mysql);
    }
}

bool Connection::execute(const char* sql)
{
    if (!m_mysql)
        return false;

    if (::mysql_query(m_mysql, sql)) {
        uint32 err_no = ::mysql_errno(m_mysql);
        FAKE_LOG_ERROR() << "mysql_query " << err_no << ":" << ::mysql_error(m_mysql);

        if (handleMySQLErrno(err_no))  // If it returns true, an error was handled successfully (i.e. reconnection)
            return execute(sql);       // Try again
        return false;
    }
    return true;
}

bool Connection::execute(MySQLPreparedStatementUPtr& stmt)
{
    if (!m_mysql)
        return false;
    MYSQL_STMT* msql_stmt = stmt->getMYSQL_STMT();
    MYSQL_BIND* msql_bind = stmt->getMYSQL_BIND();

    if (::mysql_stmt_bind_param(msql_stmt, msql_bind)) {
        uint32 err_no = ::mysql_errno(m_mysql);
        FAKE_LOG_ERROR() << "mysql_stmt_bind_param " << err_no << ":" << ::mysql_stmt_error(msql_stmt);

        if (handleMySQLErrno(err_no))  // If it returns true, an error was handled successfully (i.e. reconnection)
            return execute(stmt);       // Try again
        return false;
    }

    if (::mysql_stmt_execute(msql_stmt)) {
        uint32 err_no = ::mysql_errno(m_mysql);
        FAKE_LOG_ERROR() << "mysql_stmt_execute " << err_no << ":" << ::mysql_stmt_error(msql_stmt);

        if (handleMySQLErrno(err_no))  // If it returns true, an error was handled successfully (i.e. reconnection)
            return execute(stmt);       // Try again
        return false;
    }
    return true;
}

bool Connection::queryDetail(PreparedStatement& stmt)
{
    if (!m_mysql)
        return false;
    MYSQL_STMT* msql_stmt = stmt.getMYSQL_STMT();
    MYSQL_BIND* msql_bind = stmt.getMYSQL_BIND();
    if (::mysql_stmt_bind_param(msql_stmt, msql_bind)) {
        uint32 err_no = ::mysql_errno(m_mysql);
        FAKE_LOG_ERROR() << "mysql_stmt_bind_param " << err_no << ":" << ::mysql_stmt_error(msql_stmt);
        if (handleMySQLErrno(err_no))  // If it returns true, an error was handled successfully (i.e. reconnection)
            return queryDetail(stmt);       // Try again
        return false;
    }

    if (::mysql_stmt_execute(msql_stmt)) {
        uint32 err_no = ::mysql_errno(m_mysql);
        FAKE_LOG_ERROR() << "mysql_stmt_execute " << err_no << ":" << ::mysql_stmt_error(msql_stmt);
        if (handleMySQLErrno(err_no))  // If it returns true, an error was handled successfully (i.e. reconnection)
            return queryDetail(stmt);      // Try again
        return false;
    }
    return true;
}

bool Connection::queryDetail(const char *sql)
{
    if (!m_mysql)
        return false;

    if (::mysql_query(m_mysql, sql)) {
        uint32 err_no = ::mysql_errno(m_mysql);
        FAKE_LOG_ERROR() << "mysql_query " << err_no << ":" << ::mysql_error(m_mysql);
        if (handleMySQLErrno(err_no))      // If it returns true, an error was handled successfully (i.e. reconnection)
            return queryDetail(sql);    // We try again
        return false;
    }
    return true;
}

void Connection::beginTransaction()
{
    execute("START TRANSACTION");
}

void Connection::rollbackTransaction()
{
    execute("ROLLBACK");
}

void Connection::commitTransaction()
{
    execute("COMMIT");
}

MySQLPreparedStatementUPtr Connection::prepareStatement(const char* sql)
{
    MYSQL_STMT* stmt = ::mysql_stmt_init(m_mysql);
    if (!stmt) {
        FAKE_LOG_ERROR() << "mysql_stmt_init " << ::mysql_error(m_mysql);
        return nullptr;
    }

    if (::mysql_stmt_prepare(stmt, sql, static_cast<unsigned long>(std::strlen(sql)))) {
        FAKE_LOG_ERROR() << "mysql_stmt_prepare " << ::mysql_stmt_error(stmt);
        ::mysql_stmt_close(stmt);
        return nullptr;
    }
    return util::make_unique<PreparedStatement>(stmt);
}

bool Connection::handleMySQLErrno(uint32 err_no, uint8 attempts /*= 5*/)
{
    switch (err_no) {
    case CR_SERVER_GONE_ERROR:
    case CR_SERVER_LOST:
    case CR_INVALID_CONN_HANDLE:
    case CR_SERVER_LOST_EXTENDED: {
        if (m_mysql) {
            FAKE_LOG_ERROR() << "Lost the connection to the MySQL server!";
            ::mysql_close(getMYSQL());
            m_mysql = nullptr;
        }
        /*no break*/
    }
    case CR_CONN_HOST_ERROR: {
        FAKE_LOG_INFO() << "Attempting to reconnect to the MySQL server...";
        m_reconnecting = true;
        const uint32 err_no_ex = open();
        if (!err_no_ex) {
            FAKE_LOG_INFO() << "Successfully reconnected to " << m_conn_info.database 
                << " " << m_conn_info.host << " " << m_conn_info.port;
            m_reconnecting = false;
            return true;
        }

        if ((--attempts) == 0) {
            // Shut down the server when the mysql server isn't
            // reachable for some time
            FAKE_LOG_ERROR() << "Failed to reconnect to the MySQL server,terminating the server to prevent data corruption!";
            return false;
        } else {
            // It's possible this attempted reconnect throws 2006 at us.
            // To prevent crazy recursive calls, sleep here.
            //std::this_thread::sleep_for(std::chrono::seconds(3)); // Sleep 3 seconds
            return handleMySQLErrno(err_no_ex, attempts); // Call self (recursive)
        }
    }

                             /*
    case ER_LOCK_DEADLOCK:
            return false;    // Implemented in TransactionTask::Execute and DatabaseWorkerPool<T>::DirectCommitTransaction
    // Query related errors - skip query
    case ER_WRONG_VALUE_COUNT:
    case ER_DUP_ENTRY:
        return false;

    // Outdated table or database structure - terminate core
    case ER_BAD_FIELD_ERROR:
    case ER_NO_SUCH_TABLE:
        FAKE_LOG_ERROR() << "Your database structure is not up to date. Please make sure you've executed all queries in the sql/updates folders.";
        std::this_thread::sleep_for(std::chrono::seconds(10));
        std::abort();
        return false;
    case ER_PARSE_ERROR:
        FAKE_LOG_ERROR() << "Error while parsing SQL. Core fix required.";
        std::this_thread::sleep_for(std::chrono::seconds(10));
        std::abort();
        return false;
        */
    default:
        FAKE_LOG_ERROR() <<  "Unhandled MySQL errno:" << err_no << " Unexpected behaviour possible.";
        return false;
    }

}

uint32 Connection::getErrno() const
{
    return ::mysql_errno(m_mysql);
}

const char* Connection::getError() const
{
    return ::mysql_error(m_mysql);
}

}
