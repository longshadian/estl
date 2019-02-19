#ifndef _MYSQLCPP_CONNECTION_H
#define _MYSQLCPP_CONNECTION_H

#include <mysql.h>

#include "Types.h"

namespace mysqlcpp {

class PreparedStatement;

struct ConnectionOpt
{
	ConnectionOpt() = default;
	~ConnectionOpt() = default;
	ConnectionOpt(const ConnectionOpt& rhs) = default;
	ConnectionOpt& operator=(const ConnectionOpt& rhs) = default;
	ConnectionOpt(ConnectionOpt&& rhs) = default;
	ConnectionOpt& operator=(ConnectionOpt&& rhs) = default;

    std::string user{};
    std::string password{};
    std::string database{};
    std::string host{};
    uint32      port{3306};
};

class Connection
{
public:
    Connection(ConnectionOpt conn_opt);
    ~Connection();

    Connection(Connection const& right) = delete;
    Connection& operator=(Connection const& right) = delete;
public:

    uint32 open();
    void close();

    MySQLPreparedStatementUPtr prepareStatement(const char* sql);
    PreparedResultSetPtr query(PreparedStatement& stmt);
    ResultSetPtr query(const char* sql);

    bool execute(const char* sql);
    bool execute(MySQLPreparedStatementUPtr& stmt);
    bool queryDetail(const char *sql);
    bool queryDetail(PreparedStatement& stmt);

    void beginTransaction();
    void rollbackTransaction();
    void commitTransaction();

    operator bool () const { return m_mysql != NULL; }
    void ping() { ::mysql_ping(m_mysql); }
    MYSQL* getMYSQL()  { return m_mysql; }

    uint32 getErrno() const;
    const char* getError() const;
private:
    bool handleMySQLErrno(uint32 err_no, uint8 attempts = 5);
private:
    bool			m_reconnecting;
    MYSQL*          m_mysql;
    ConnectionOpt	m_conn_info;
};

}

#endif
