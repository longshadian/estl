#pragma once

#include <string>

#include <mysql.h>

#include "mysqlcpp/Types.h"

namespace mysqlcpp {

class PreparedStatement;
class Statement;

struct MYSQLCPP_EXPORT ConnectionOpt
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
    std::string charset{"utf8mb4"};
};

class MYSQLCPP_EXPORT Connection
{
    friend class Statement;
    friend class PreparedStatement;

public:
    Connection(ConnectionOpt conn_opt);
    ~Connection();

    Connection(Connection const& right) = delete;
    Connection& operator=(Connection const& right) = delete;
public:
    bool init();
    bool open();
    void close();

    operator bool () const { return m_mysql != NULL; }
    int ping();
    MYSQL* getMYSQL();
    uint32 getErrorNo() const;
    const std::string& getErrorStr() const;
    void clearError();

    StatementPtr statement();
    PreparedStatementPtr preparedStatement(const char* sql);
    PreparedStatementPtr preparedStatement(const std::string& sql);

private:
    void storeError(uint32 err_no, const char* err_str);
    PreparedStatementPtr createPreparedStatement(const char* sql, size_t sql_len);
private:
    MYSQL*          m_mysql;
    ConnectionOpt	m_conn_info;
    uint32          m_err_no;
    std::string     m_err_str;
};

}
