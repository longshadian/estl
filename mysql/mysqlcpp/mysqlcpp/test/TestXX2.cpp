#include <thread>
#include <chrono>
#include <iostream>
#include <memory>

#include "mysqlcpp/mysqlcpp.h"

mysqlcpp::ConnectionOpt initOpt()
{
    mysqlcpp::ConnectionOpt conn_opt{};
    conn_opt.user = "root";
    conn_opt.password = "123456";
    conn_opt.database = "my_test";
    conn_opt.host = "192.168.207.128";
    conn_opt.port = 3306;
    return conn_opt;
}

int fun()
{
    // 创建数据库连接
    mysqlcpp::ConnectionOpt conn_opt = initOpt();
    mysqlcpp::Connection db_conn{ conn_opt };
    if (db_conn.open() != 0) {
        //LOG(WARNING) << "create db conn";
        return 0;
    }

    auto stmt = db_conn.statement();
    if (!stmt) {
        std::cout << "create stmt fail\n";
        return 0;
    }

    for (int i = 0; i != 1; ++i) {
        auto s = std::to_string(i*2);
        std::array<char, 1024> buffer{};
        snprintf(buffer.data(), buffer.size(),
            "INSERT INTO `my_test`.`test_string` "
            " ( `uid`, `name`) VALUES (%d, %s)"
            , i, s.c_str()
            );

        if (!stmt->execute(buffer.data())) {
            std::cout << "execute fail\n";
            return 0;
        }
    }

    const char* sql = "SELECT `uid`, `name` from `test_string` WHERE `id` = ? ";
    auto ps = db_conn.preparedStatement(sql);
    if (!ps) {
        std::cout << "create preparedStatement\n";
        return 0;
    }

    for (int i = 51; i != 61; ++i) {
        ps->clearParameters();
        ps->setInt32(0, i);
        auto rs = ps->executeQuery();
        if (!rs) {
            std::cout << "ps query fail\n";
            return 0;
        }
        std::cout << "row count:" << rs->getRowCount() << "\t";
        for (uint64_t i = 0; i != rs->getRowCount(); ++i) {
            auto row = rs->getRow(i);
            std::cout << row["uid"]->getInt32() << " " << row["name"]->getString() << "\n";
        }
    }

    return 0;
}

int funStmt()
{
    // 创建数据库连接
    mysqlcpp::ConnectionOpt conn_opt = initOpt();
    mysqlcpp::Connection db_conn{ conn_opt };
    if (db_conn.open() != 0) {
        //LOG(WARNING) << "create db conn";
        return 0;
    }

    auto stmt = db_conn.statement();
    auto rs = stmt->executeQuery("select uid, name from test_string where id = 2");
    if (!rs) {
        std::cout << "rs null\n";
        return 0;
    }

    for (uint64_t i = 0; i != rs->getRowCount(); ++i) {
        auto row = rs->getRow(i);
        std::cout << row["uid"]->getInt32() 
            << "\t" << row["name"]->isNull() 
            << "\t" << row["name"]->getString() 
            << "\n";
    }

    std::cout << __FUNCTION__ << "\n";
    return 0;
}

int funPreparedStmt()
{
    // 创建数据库连接
    mysqlcpp::ConnectionOpt conn_opt = initOpt();
    mysqlcpp::Connection db_conn{ conn_opt };
    if (db_conn.open() != 0) {
        //LOG(WARNING) << "create db conn";
        return 0;
    }

    const char* sql = 
        " update test_string set uid = ?, name = ? where ? <= id and id <= ?";
    auto ps = db_conn.preparedStatement(sql);
    if (!ps) {
        std::cout << "preparedStatement fail\n";
        return 0;
    }

    for (int i = 0; i != 10; ++i) {
        ps->clearParameters();
        ps->setInt32(0, i);
        ps->setString(1, "xx");
        ps->setInt32(2, i * 1);
        ps->setInt32(3, i * 2);

        /*
        auto rs = ps->executeQuery();
        if (!rs)
            std::cout << "rs null\n";
        */
        if (!ps->execute()) {
            std::cout << "ps execute fail\n";
        } else {
            std::cout << "ps execute affect rows " << ps->getAffectedRows() << "\n";
        }
    }

    std::cout << __FUNCTION__ << "\n";
    return 0;
}

int testTransaction()
{
    // 创建数据库连接
    mysqlcpp::ConnectionOpt conn_opt = initOpt();
    mysqlcpp::Connection db_conn{ conn_opt };
    if (db_conn.open() != 0) {
        return 0;
    }

    {
        mysqlcpp::Transaction ts{db_conn};
        auto stmt = db_conn.statement();
        const char* sql = "delete from test_string where id = 1";
        stmt->execute(sql);

        sql = "update test_string set uid = ?, name = ? where id = ?";
        auto ps = db_conn.preparedStatement(sql);
        if (!ps) {
            std::cout << "ps error\n";
            return 0;
        }
        ps->setInt32(0, 999);
        ps->setString(1, "aaaaa");
        ps->setInt32(2, 2);
        if (ps->execute()) {
            ts.commit();
            std::cout << "success\n";
        }
    }
    std::cout << __FUNCTION__ << "\n";
    return 0;
}

int testPool()
{
    // 创建数据库连接
    mysqlcpp::ConnectionOpt conn_opt = initOpt();

    mysqlcpp::ConnectionPoolOpt pool_opt{};
    pool_opt.m_thread_pool_size = 3;

    mysqlcpp::ConnectionPool pool{conn_opt, pool_opt};
    if (!pool.init()) {
        std::cout << "pool init fail\n";
        return 0;
    }

    {
        mysqlcpp::ConnectionGuard guard{ pool };
        auto stmt = guard->statement();
        auto rs = stmt->executeQuery("select 2");
        std::cout << "select " << rs->getRow(0)["2"]->getString() << "\n";
    }

    int n = 10;
    while (n > 0) {
        std::cout << --n << "\n";
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    {
        mysqlcpp::ConnectionGuard guard{ pool };
        auto stmt = guard->statement();
        auto rs = stmt->executeQuery("select 3");
        std::cout << "select " << rs->getRow(0)["3"]->getString() << "\n";
    }

    std::cout << __FUNCTION__ << "\n";
    return 0;
}

int main()
{
    //zylib::logger::initSyncConsole();
    //mysqlcpp::initLog(&std::cout);

    //LOG(DEBUG) << "aaa from zylog";
    //BOOST_LOG_TRIVIAL(debug) << "sssss";

    //funPreparedStmt();
    //funStmt();
    //testTransaction();
    testPool();

    system("pause");
    return 0;
}
