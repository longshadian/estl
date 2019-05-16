#include <thread>
#include <chrono>
#include <iostream>
#include <memory>

#include "mysqlcpp/mysqlcpp.h"

bool selectMinMaxUserID(mysqlcpp::Connection& conn, uint64_t* min_id, uint64_t* max_id);
bool selectMinMaxUserIDEx(mysqlcpp::Connection& conn, uint64_t* min_id, uint64_t* max_id);

int fun(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    // 创建数据库连接
    mysqlcpp::ConnectionOpt conn_opt{};
    conn_opt.user = "root";
    conn_opt.password = "123456";
    conn_opt.database = "my_test";
    conn_opt.host = "192.168.207.128";
    conn_opt.port = 3306;

    mysqlcpp::Connection db_conn{ conn_opt };
    if (db_conn.open() != 0) {
        //LOG(WARNING) << "create db conn";
        return 0;
    }

    uint64_t max_id = 0;
    uint64_t min_id = 0;
    (void)max_id;
    (void)min_id;

    if (!selectMinMaxUserID(db_conn, &min_id, &max_id)) {
        return 0;
    }

    const char* sql = "SELECT `id`, `uid`, `name` from `test_string` WHERE ? <= `id` AND `id` < ? ";
    auto stmt = db_conn.preparedStatement(sql);

    uint64_t pos = min_id;
    const uint64_t LIMIT = 10;
    while (true) {
        auto pos_end = pos + LIMIT;
        stmt->clearParameters();
        stmt->setUInt64(0, pos);
        stmt->setUInt64(1, pos_end);

        auto rs = stmt->executeQuery();

        for (uint64_t i = 0; i != rs->getRowCount(); ++i) {
            auto row = rs->getRow(i);
            std::cout << row["id"]->getInt32() 
                << " " << row["uid"]->getInt32() 
                << " " << row["name"]->getString() << "\n";
        }

        if (max_id <= pos) {
            break;
        }
        pos = pos_end;
    }
    system("pause");
    return 0;
}

bool selectMinMaxUserID(mysqlcpp::Connection& conn, uint64_t* min_id, uint64_t* max_id)
{
    const char* sql = "SELECT MAX(id) AS max_id, MIN(id) as min_id FROM test_string";
    auto stmt = conn.statement();
    if (!stmt) {
        std::cout << "mysqlcpp query error:" << conn.getErrorNo() << ":" << conn.getErrorStr();
        return false;
    }
    auto rs = stmt->executeQuery(sql);
    auto row = rs->getRow(0);
    if (row["max_id"]->isNull()) {
        *max_id = 0;
    } else {
        *max_id = row["max_id"]->getUInt64();
    }

    if (row["min_id"]->isNull()) {
        *min_id = 0;
    } else {
        *min_id = row["min_id"]->getUInt64();
    }
    return true;
}

bool selectMinMaxUserIDEx(mysqlcpp::Connection& conn, uint64_t* min_id, uint64_t* max_id)
{
    const char* sql = "SELECT MAX(id) AS max_id, MIN(id) as min_id FROM test_string where 1 = ?";
    auto ps = conn.preparedStatement(sql);
    if (!ps) {
        std::cout << "preparedStatement fail\n";
        return false;
    }

    ps->setInt32(0, 1);
    auto rs = ps->executeQuery();
    auto row = rs->getRow(0);
    if (row["max_id"]->isNull()) {
        *max_id = 0;
    } else {
        *max_id = row["max_id"]->getUInt64();
    }

    if (row["min_id"]->isNull()) {
        *min_id = 0;
    } else {
        *min_id = row["min_id"]->getUInt64();
    }
    return true;
}
