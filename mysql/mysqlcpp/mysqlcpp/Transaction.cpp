#include "mysqlcpp/Transaction.h"

#include <algorithm>

#include "mysqlcpp/Connection.h"
#include "mysqlcpp/Statement.h"

namespace mysqlcpp {

Transaction::Transaction(Connection& conn)
    : m_conn(conn)
    , m_stmt(conn.statement())
    , m_rollback(true)
{
    m_stmt->execute("START TRANSACTION");
}

Transaction::~Transaction()
{
    if (m_rollback) {
        m_stmt->execute("ROLLBACK");
    }
}

void Transaction::commit()
{
    m_stmt->execute("COMMIT");
    m_rollback = false;
}


}
