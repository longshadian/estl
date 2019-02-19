#include "ConnectionPool.h"

#include <algorithm>

#include "Assert.h"
#include "FakeLog.h"

namespace mysqlcpp {

ConnectionGuard::ConnectionGuard(ConnectionPool& pool)
    : m_pool(pool)
    , m_conn(nullptr)
{
}

ConnectionGuard::~ConnectionGuard()
{
	m_pool.rleaseConn(std::move(m_conn));
}

ConnectionPool::ConnectionPool(ConnectionOpt conn_opt, ConnectionPoolOpt pool_opt)
	: m_mutex()
	, m_pool()
	, m_conn_opt(std::move(conn_opt))
	, m_pool_opt(std::move(pool_opt))
{

}

bool ConnectionPool::init()
{
	for (size_t i = 0; i != m_pool_opt.m_thread_pool_size; ++i) {
		auto conn = std::make_shared<Connection>(m_conn_opt);
		if (conn->open() != 0) {
			FAKE_LOG_ERROR() << "create mysql conn error";
			return false;
		}
		m_pool.push_back({conn, std::time(nullptr), false});
	}
	return true;
}

std::shared_ptr<Connection> ConnectionPool::getConn()
{
    {
		std::lock_guard<std::mutex> lk{ m_mutex };
		Slot* slot = findEmptySlot();
		if (slot) {
			slot->m_in_use = true;
			return slot->m_conn;
		}

		if (m_pool.size() >= m_pool_opt.m_thread_pool_max_threads) {
			FAKE_LOG_WARRING() << "too much connection! count:" << m_pool.size();
			return nullptr;
		}
    }

	//创建新数据库链接
	auto conn = create();
	if (!conn) {
		FAKE_LOG_ERROR() << "can't create new mysql connection error!";
		return nullptr;
	}
	Slot new_slot{conn, std::time(nullptr), true};
	{
		//TODO 再次检测链接数量
		std::lock_guard<std::mutex> lk{ m_mutex };
		m_pool.push_back(new_slot);
	}
	return new_slot.m_conn;
}

std::shared_ptr<Connection> ConnectionPool::create()
{
    auto conn = std::make_shared<Connection>(m_conn_opt);
    if (conn->open() == 0)
        return conn;
    return nullptr;
}

void ConnectionPool::rleaseConn(std::shared_ptr<Connection> conn)
{
	std::lock_guard<std::mutex> m_lk{ m_mutex };
	auto tnow = std::time(nullptr);

    auto* slot = findSlot(conn);
    ASSERT(slot);
    ASSERT(slot->m_in_use);
    slot->m_in_use = false;
	slot->m_last_used = tnow;

	//销毁超时链接
	destoryTimeout(tnow);
}

ConnectionPool::Slot* ConnectionPool::findSlot(const std::shared_ptr<Connection>& conn)
{
    auto it = std::find_if(m_pool.begin(), m_pool.end(), [&conn](const Slot& p) { return p.m_conn == conn; });
    if (it != m_pool.end())
        return &*it;
    return nullptr;
}

ConnectionPool::Slot* ConnectionPool::findEmptySlot()
{
    for (auto& s : m_pool) {
        if (!s.m_in_use)
            return &s;
    }
    return nullptr;
}

void ConnectionPool::destoryTimeout(time_t tnow)
{
	for (auto it = m_pool.begin(); it != m_pool.end();) {
		const Slot& s = *it;
		if (s.m_in_use)
			continue;
		if (tnow - s.m_last_used >= static_cast<time_t>(m_pool_opt.m_thread_pool_idle_timeout)) {
			it = m_pool.erase(it);
		} else {
			++it;
		}
	}
}

}

