#include <cassert>
#include <ctime>
#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <queue>
#include <future>
#include <iostream>

#include "mysqlcpp/mysqlcpp.h"
#include "AsyncTask.h"

using PoolPtr = std::shared_ptr<mysqlcpp::ConnectionPool>;
const size_t POOL_SIZE = 4;

mysqlcpp::ConnectionOpt initConn();
PoolPtr initPool();

class DBService
{
public:
    DBService()
        : m_mtx()
        , m_cond()
        , m_queue()
        , m_running()
        , m_threads()
        , m_db_pool()
    {
    }

    ~DBService()
    {
        stop();
        waitTheadExit();
    }

    DBService(const DBService& rhs) = delete;
    DBService& operator=(const DBService& rhs) = delete;

    bool init(size_t num_threads)
    {
        if (m_running.exchange(true))
            return false;
        m_db_pool = initPool();
        if (!m_db_pool) {
            return false;
        }
        for (size_t i = 0; i != num_threads; ++i) {
            m_threads.push_back(std::thread(&DBService::threadStart, this));
        }
        return true;
    }

    void stop()
    {
        m_running.exchange(false);
    }

    void waitTheadExit()
    {
        for (auto& thread : m_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    size_t remainTask() const
    {
        std::lock_guard<std::mutex> lk{ m_mtx };
        return m_queue.size();
    }

    template<typename F>
    std::future<typename std::result_of<F()>::type> asyncSubmit(F f)
    {
        typedef typename std::result_of<F()>::type result_type;
        std::packaged_task<result_type()> task(std::move(f));
        auto res = task.get_future();
        std::lock_guard<std::mutex> lk(m_mtx);
        m_queue.push(std::move(task));
        m_cond.notify_all();
        return res;
    }

    std::future<int> testCreateTable()
    {
        return asyncSubmit([this]
        {
            mysqlcpp::ConnectionGuard conn{ *m_db_pool };
            assert(conn);
            const char* sql_threads =
                "   CREATE TABLE `mysqlcpp_test`.`test_threads` ("
                "   `fpk` INT NOT NULL AUTO_INCREMENT,"
                "   `fint` INT(11) NULL,"
                "   `fbigint` BIGINT(20) NULL,"
                "   PRIMARY KEY(`fpk`))"
                "   ENGINE = InnoDB"
                "   DEFAULT CHARACTER SET = utf8";
            auto stmt = conn->statement();
            assert(stmt->execute("DROP TABLE IF EXISTS test_threads;"));
            assert(stmt->execute(sql_threads));
            return 1;
        });
    }

    void testInsert(int32_t v1, int64_t v2)
    {
        asyncSubmit([this, v1, v2] 
        {
            mysqlcpp::ConnectionGuard conn{ *m_db_pool };
            assert(conn);
            const char* sql = "INSERT INTO `test_threads` (`fint`, `fbigint`) "
                " VALUES (?, ?)";
            auto ps = conn->preparedStatement(sql);
            assert(ps);
            ps->setInt32(0, v1);
            ps->setInt64(1, v2);
            assert(ps->execute());
            std::ostringstream ostm{};
            ostm << "execute thread " << conn.get() << " " << std::this_thread::get_id() << "\n";
            std::cout << ostm.str();
            std::this_thread::sleep_for(std::chrono::milliseconds{ 10 });
        });
    }

    void testQuery(int32_t val)
    {
        asyncSubmit([this, val]
        {
            mysqlcpp::ConnectionGuard conn{ *m_db_pool };
            assert(conn);

            const char* sql = "select fint, fbigint from `test_threads` where fpkid = ?";
            auto ps = conn->preparedStatement(sql);
            assert(ps);
            ps->setInt32(0, val);
            auto rs = ps->executeQuery();
            assert(rs);
            if (rs->getRowCount() != 0) {
                auto row = rs->getRow(0);
                std::cout << row["fint"]->getInt32() << " " << row["fbigint"]->getInt64();
            }
            std::cout << "  execute thread " << conn.get() << "\n";
        });
    }

private:
    void threadStart()
    {
        try {
            run();
        }
        catch (...) {
        }
    }

    void run()
    {
        while (m_running) {
            zylib::AsyncTask task{};
            {
                std::unique_lock<std::mutex> lk(m_mtx);
                m_cond.wait_for(lk, std::chrono::seconds{2}, [this] { return !m_queue.empty(); });
                if (!m_queue.empty()) {
                    task = std::move(m_queue.front());
                    m_queue.pop();
                }
            }
            if (task) {
                task();
            }
        }
    }


private:
    mutable std::mutex	        m_mtx;
    std::condition_variable     m_cond;
    std::queue<zylib::AsyncTask> m_queue;
    std::atomic<bool>           m_running;
    std::vector<std::thread>    m_threads;
    std::shared_ptr<mysqlcpp::ConnectionPool> m_db_pool;
};

mysqlcpp::ConnectionOpt initConn()
{
    mysqlcpp::ConnectionOpt conn_opt{};
    conn_opt.user = "root";
    conn_opt.password = "123456";
    conn_opt.database = "mysqlcpp_test";
    conn_opt.host = "127.0.0.1";
    conn_opt.port = 3306;
    return conn_opt;
}

PoolPtr initPool()
{
    mysqlcpp::ConnectionPoolOpt pool_opt{};
    pool_opt.m_thread_pool_size = POOL_SIZE;
    mysqlcpp::ConnectionOpt conn_opt = initConn();

    auto pool = std::make_shared<mysqlcpp::ConnectionPool>(conn_opt, pool_opt);
    if (!pool->init()) {
        std::cout << "pool init fail\n";
        return nullptr;
    }
    return pool;
}

int main()
{
    std::srand((unsigned int)std::time(nullptr));
    mysqlcpp::initLog(std::make_unique<mysqlcpp::LogStreamConsole>());

    DBService service{};
    if (!service.init(POOL_SIZE)) {
        std::cout << "service init fail\n";
        return 0;
    }

    auto f = service.testCreateTable();
    if (f.get() != 1) {
        std::cout << "get fail\n";
        return 0;
    }

    for (int32_t i = 0; i != 100; ++i) {
        auto v1 = std::rand() % 10000;
        auto v2 = std::rand() % 10000;
        service.testInsert(v1, v2);
    }

    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds{ 1 });
        auto n = service.remainTask();
        if (n == 0)
            break;
        std::cout << "remain task " << n << "\n";
    }
    std::this_thread::sleep_for(std::chrono::seconds{ 1 });

    return 0;
}
