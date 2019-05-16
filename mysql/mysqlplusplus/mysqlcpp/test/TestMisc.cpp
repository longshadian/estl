
#include <ctime>
#include <iostream>
#include <string>
#include <sys/time.h>

#include "MySqlCpp.h"

mysqlcpp::ConnectionOpt initConn()
{
    mysqlcpp::ConnectionOpt conn_info{};
    conn_info.user = "root";
    conn_info.password = "123456";
    conn_info.database = "mytest";
    conn_info.host = "127.0.0.1";
    conn_info.port = 3306;
    return conn_info;
}

int test()
{
    auto conn_info = initConn();
    mysqlcpp::Connection conn{conn_info};
    if (conn.open() != 0) {
        std::cout << "open error\n";
        return 0;
    }

    const std::string sql = "select fid, fname as xx, fsid from test";
    auto ret = conn.query(sql.c_str());
    if (!ret) {
        std::cout << "query error\n";
        return 0;
    }

    for (uint32_t i = 0; i != ret->getRowCount(); ++i) {
        const auto& row = ret->getRow(i);
        std::cout << "row:" << row[0].getInt16() << "\t" << row[1].getString() << "\t" << row[2].getInt32() << "\n";
    }
    return 0;
}

int testStmt()
{
    auto conn_info = initConn();
    mysqlcpp::Connection conn{conn_info};
    if (conn.open() != 0) {
        std::cout << "open error\n";
        return 0;
    }

    const std::string sql = "select fid, fname, fsid from test where fid = ?";
    auto stmt = conn.prepareStatement(sql.c_str());
    if (!stmt) {
        std::cout << "prepareStatement error\n";
        return 0;
    }

    stmt->setInt16(0, 1);
    auto ret = conn.query(*stmt);
    if (!ret) {
        std::cout << "query error\n";
        return 0;
    }

    //std::cout << ret->getRowCount() << " " << ret->fetch()[1].getString() << "\n";
    for (uint32_t i = 0; i != ret->getRowCount(); ++i) {
        const auto& row = ret->getRow(i);
        std::cout << "row:" << row[0].getInt16() << "\t" << row[1].getString() << "\t" << row[2].getInt32() << "\n";
    }
    return 0;
}

int testStmtUpdate()
{
    auto conn_info = initConn();
    mysqlcpp::Connection conn{conn_info};
    if (conn.open() != 0) {
        std::cout << "open error\n";
        return 0;
    }

    const std::string sql = "update test set fname=?, ftime=?, fdate=?, fdatetime = ?, ftimestamp=? where fid = ?";
    auto stmt = conn.prepareStatement(sql.c_str());
    if (!stmt) {
        std::cout << "prepareStatement error\n";
        return 0;
    }

    //auto tnow = std::time(nullptr);
    timeval tv;
    gettimeofday(&tv, nullptr);
    //stmt->setString(0, "hesshe");
    stmt->setNull(0);
    stmt->setDateTime(1, mysqlcpp::DateTime(tv));
    stmt->setDateTime(2, mysqlcpp::DateTime(tv));
    stmt->setDateTime(3, mysqlcpp::DateTime());
    stmt->setDateTime(4, mysqlcpp::DateTime("12:57:21"));
    stmt->setInt32(5, 1);
    auto ret = conn.query(*stmt);
    if (!ret) {
        std::cout << "query error\n";
        return 0;
    }
    return 0;
}

int testTime()
{
    auto conn_info = initConn();
    mysqlcpp::Connection conn{conn_info};
    if (conn.open() != 0) {
        std::cout << "open error\n";
        return 0;
    }

    const std::string sql = "select fdatetime, ftime, fdate, ftimestamp, year(fdatetime), month(fdatetime), date(fdate) from test where fid = 1";
    auto ret = conn.query(sql.c_str());
    if (!ret) {
        std::cout << "query error\n";
        return 0;
    }
    for (uint32_t i = 0; i != ret->getRowCount(); ++i) {
        const auto& row = ret->getRow(i);
        std::cout << "row:" << row[0].getString() << "\t" << row[0].getDateTime().getTime() << "\n";
        std::cout << "row:" << row[1].getString() << "\t" << row[1].getDateTime().getTime() << "\n";
        std::cout << "row:" << row[2].getString() << "\t" << row[2].getDateTime().getTime() << "\n";
        std::cout << "row:" << row[3].getString() << "\t" << row[3].getDateTime().getTime() << "\n";
    }
    return 0;
}

int main()
{
    //test();
    std::cout << "\n\n";
    //testStmt();
    //testStmtUpdate();
    testTime();

    return 0;
}
