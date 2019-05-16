#pragma once

#include <memory>
#include <mysql.h>

#include "mysqlcpp/Types.h"

namespace mysqlcpp {

class Connection;

class MYSQLCPP_EXPORT Transaction
{
public:
    Transaction(Connection& conn);
    ~Transaction();
    Transaction(const Transaction& rhs) = delete;
    Transaction& operator=(const Transaction& rhs) = delete;

    void commit();
private:
    Connection&  m_conn;
    StatementPtr m_stmt;
    bool         m_rollback;
};

}
