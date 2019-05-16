#pragma once

#include <ctime>
#include <list>
#include <mutex>

#include "mysqlcpp/Types.h"
#include "mysqlcpp/Connection.h"

namespace mysqlcpp {

class Connection;
class ConnectionPool;

class MYSQLCPP_EXPORT ConnectionGuard
{
public:
    ConnectionGuard(ConnectionPool& pool);
    ~ConnectionGuard();

	Connection& operator*() const { return *m_conn; }
	const std::shared_ptr<Connection>& operator->() const { return m_conn; }
	operator bool() const { return m_conn != nullptr; }

	Connection* get() const { return m_conn.get(); }
private:
    ConnectionPool&  m_pool;
	std::shared_ptr<Connection> m_conn;
};

struct MYSQLCPP_EXPORT ConnectionPoolOpt
{
	ConnectionPoolOpt() = default;
	~ConnectionPoolOpt() = default;
	ConnectionPoolOpt(const ConnectionPoolOpt& rhs) = default;
	ConnectionPoolOpt& operator=(const ConnectionPoolOpt& rhs) = default;
	ConnectionPoolOpt(ConnectionPoolOpt&& rhs) = default;
	ConnectionPoolOpt& operator=(ConnectionPoolOpt&& rhs) = default;

	size_t m_thread_pool_size{3};           //线程池初始线程个数
};

class MYSQLCPP_EXPORT ConnectionPool
{
    struct Slot
    {
        Slot(std::shared_ptr<Connection> conn , time_t t, bool use)
            : m_conn(conn)
            , m_last_used(t)
            , m_in_use(use)
        {
        }

		std::shared_ptr<Connection>		m_conn;
        time_t                          m_last_used;
        bool                            m_in_use;
    };
    using SlotPtr = std::shared_ptr<Slot>;

public:
	ConnectionPool(ConnectionOpt conn_opt, ConnectionPoolOpt pool_opt = {});
    ~ConnectionPool();

    ConnectionPool(const ConnectionPool& rhs) = delete;
	ConnectionPool& operator=(const ConnectionPool& rhs) = delete;

	bool init();
    std::shared_ptr<Connection> getConn();
    void rleaseConn(std::shared_ptr<Connection> conn);
    size_t connectionCount() const;
private:
    std::shared_ptr<Connection> create() const;
    SlotPtr findEmptySlot();
private:
    mutable std::mutex      m_mutex;
    std::list<SlotPtr>      m_pool;
    ConnectionOpt			m_conn_opt;
	ConnectionPoolOpt		m_pool_opt;
};

}
