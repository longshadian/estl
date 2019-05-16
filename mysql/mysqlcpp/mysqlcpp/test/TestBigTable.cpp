#include <thread>
#include <chrono>
#include <iostream>
#include <memory>

#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "mysqlcpp/mysqlcpp.h"

std::shared_ptr<boost::uuids::random_generator> g_generator;

using PoolPtr = std::shared_ptr<mysqlcpp::ConnectionPool>;
const int32_t POOL_SIZE = 4;

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
    mysqlcpp::ConnectionOpt conn_opt = initConn();

    auto pool = std::make_shared<mysqlcpp::ConnectionPool>(conn_opt, pool_opt);
    if (!pool->init()) {
        std::cout << "pool init fail\n";
        return nullptr;
    }
    return pool;
}

std::string newUUID()
{
    auto& m_generator = *g_generator;
    boost::uuids::uuid u = m_generator();
    return boost::uuids::to_string(u);
}

std::vector<uint8_t> newBlob()
{
    std::vector<uint8_t> val;
    val.resize(1024 * 60, 'a');
    return val;
}

mysqlcpp::ConnectionOpt initOpt()
{
    mysqlcpp::ConnectionOpt conn_opt{};
    conn_opt.user = "root";
    conn_opt.password = "123456";
    conn_opt.database = "test";
    conn_opt.host = "192.168.207.128";
    conn_opt.port = 3306;
    return conn_opt;
}

int initBigTable()
{
    // 创建数据库连接
    auto pool = initPool();
    if (!pool) {
        std::cout << "init pool error\n";
        return 0;
    }
    mysqlcpp::ConnectionGuard conn{ *pool };
    auto fblob = newBlob();
    const char* sql =
        " INSERT INTO `test`.`test_big_table` (`fname`, `fblob`) "
        " VALUES ( ? , ?) ";
    /*
    auto ps = conn->preparedStatement(sql);
    if (!ps) {
        std::cout << "preparedStatement fail\n";
        return 0;
    }
    */

    {
        for (int j = 0; j != 1000; ++j) {
            auto tnow = std::chrono::system_clock::now();
            mysqlcpp::Transaction tran{ *conn };
            for (int i = 0; i != 1000; ++i) {
                auto ps = conn->preparedStatement(sql);
                if (!ps) {
                    std::cout << "preparedStatement fail\n";
                    return 0;
                }

                ps->clearParameters();
                ps->setString(0, newUUID());
                ps->setBinary(1, fblob, false);

                if (!ps->execute()) {
                    std::cout << "ps execute fail\n";
                    break;
                }
            }
            tran.commit();
            auto tend = std::chrono::system_clock::now();
            std::cout << j << "  cost:" << std::chrono::duration_cast<std::chrono::seconds>(tend - tnow).count() << "\n";
        }
    }

    //auto tend = std::chrono::system_clock::now();
    //std::cout << "cost:" << std::chrono::duration_cast<std::chrono::seconds>(tend - tnow).count() << "\n";
    return 0;
}

int main()
{
    g_generator = std::make_shared<boost::uuids::random_generator>();
    initBigTable();
    return 0;
}
