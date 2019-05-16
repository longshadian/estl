#include "mysqlcpp/ConnectionPool.h"

#include <mysql.h>
#include <algorithm>

#include "mysqlcpp/MysqlcppAssert.h"
#include "mysqlcpp/MysqlcppLog.h"
#include "mysqlcpp/Statement.h"

namespace mysqlcpp {

ConnectionGuard::ConnectionGuard(ConnectionPool& pool)
    : m_pool(pool)
    , m_conn(nullptr)
{
	m_conn = m_pool.getConn();
}

ConnectionGuard::~ConnectionGuard()
{
	if (m_conn) {
		m_pool.rleaseConn(std::move(m_conn));
	}
}

ConnectionPool::ConnectionPool(ConnectionOpt conn_opt, ConnectionPoolOpt pool_opt)
	: m_mutex()
	, m_pool()
	, m_conn_opt(std::move(conn_opt))
	, m_pool_opt(std::move(pool_opt))
{

}

ConnectionPool::~ConnectionPool()
{
    ::mysql_library_end();
}

bool ConnectionPool::init()
{
    if (::mysql_library_init(0, nullptr, nullptr)) {
        MYSQLCPP_LOG(Error, "mysql_library_init fail");
        return false;
    }

	for (size_t i = 0; i != m_pool_opt.m_thread_pool_size; ++i) {
        auto conn = create();
		if (!conn) {
            MYSQLCPP_LOG(Error, "create mysql conn error");
			return false;
		}
        auto slot = std::make_shared<Slot>(conn, std::time(nullptr), false);
		m_pool.push_back(slot);
	}
	return true;
}

std::shared_ptr<Connection> ConnectionPool::getConn()
{
    std::lock_guard<std::mutex> lk{ m_mutex };
    auto slot = findEmptySlot();
    if (slot) {
        slot->m_in_use = true;
        return slot->m_conn;
    }
    return nullptr;
}

std::shared_ptr<Connection> ConnectionPool::create() const
{
    auto conn = std::make_shared<Connection>(m_conn_opt);
    if (!conn->init())
        return nullptr;
    if (!conn->open())
        return nullptr;
    return conn;
}

void ConnectionPool::rleaseConn(std::shared_ptr<Connection> conn)
{
    conn->clearError();
	auto tnow = std::time(nullptr);
    {
        std::lock_guard<std::mutex> m_lk{ m_mutex };

        auto it = std::find_if(m_pool.begin(), m_pool.end(), [&conn](const SlotPtr& p) { return p->m_conn == conn; });
        MYSQLCPP_ASSERT(it != m_pool.end());
        auto slot = *it;
        slot->m_in_use = false;
        slot->m_last_used = tnow;
        // �ŵ��б�����
        m_pool.erase(it);
        m_pool.push_back(slot);
    }
}

size_t ConnectionPool::connectionCount() const
{
    std::lock_guard<std::mutex> lk{ m_mutex };
    return m_pool.size();
}

ConnectionPool::SlotPtr ConnectionPool::findEmptySlot()
{
    for (auto& s : m_pool) {
        if (!s->m_in_use)
            return s;
    }
    return nullptr;
}

} // mysqlcpp
