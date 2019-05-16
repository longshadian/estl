#include "Transaction.h"

#include <algorithm>

#include "Connection.h"

namespace mysqlcpp {

Transaction::Transaction(Connection& conn)
    : m_conn(conn)
    , m_rollback(true)
{
    m_conn.beginTransaction();
}

Transaction::~Transaction()
{
    if (m_rollback)
        m_conn.rollbackTransaction();
}

void Transaction::commit()
{
    m_rollback = false;
    m_conn.commitTransaction();
}


}
