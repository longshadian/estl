
#include <cassert>
#include <ctime>
#include <iostream>
#include <string>
#include <memory>
#include <sys/time.h>

#include "MySqlCpp.h"

#define ASSERT assert

#define CREATE_SCHEMA

mysqlcpp::ConnectionOpt initConn()
{
    mysqlcpp::ConnectionOpt conn_info{};
    conn_info.user = "root";
    conn_info.password = "123456";
    //conn_info.database = "mytest";
    conn_info.host = "127.0.0.1";
    conn_info.port = 3306;
    return conn_info;
}

void test()
{
    auto conn_info = initConn();
    mysqlcpp::Connection conn{conn_info};
    if (conn.open() != 0) {
        std::cout << "open error\n";
        return;
    }
    std::shared_ptr<mysqlcpp::ResultSet> ret;
    std::shared_ptr<mysqlcpp::PreparedStatement> stmt;
    const char* sql = nullptr;

#ifdef CREATE_SCHEMA
    sql = "CREATE DATABASE IF NOT EXISTS mysqlcpp_test DEFAULT CHARACTER SET utf8";
    //sql = "create schema  mysqlcpp_test default character set utf8;";
    ret = conn.query(sql);
    //ret = conn.query(sql);
    ASSERT(ret && ret->getRowCount() == 0);
#endif
    ret = conn.query("use mysqlcpp_test;");
    ASSERT(ret && ret->getRowCount() == 0);

    ret = conn.query("DROP TABLE IF EXISTS test;");
    ASSERT(ret && ret->getRowCount() == 0);

    sql = 
    "   CREATE TABLE `mysqlcpp_test`.`test` ("
    "   `pk` INT NOT NULL,"
    "   `ftinyint` TINYINT(11) NULL,"
    "   `fsmallint` SMALLINT(11) NULL,"
    "   `fint` INT(11) NULL,"
    "   `fbigint` BIGINT(20) NULL,"
    "   `fchar` CHAR(10) NULL,"
    "   `fvarchar` VARCHAR(10) NULL,"
    "   `ffloat` FLOAT(10, 2) NULL,"
    "   `fdouble` DOUBLE(10, 4) NULL,"
    "   `fdecimal` DECIMAL(10, 6) NULL,"
    "   `fdate` DATE NULL,"
    "   `ftime` TIME NULL,"
    "   `fdatetime` DATETIME NULL,"
    "   `ftimestamp` TIMESTAMP NULL,"
    "   `fblob` BLOB NULL,"
    "   `fbinary` BINARY NULL,"
    "   `ftext` TEXT NULL,"
    "   PRIMARY KEY(`pk`))"
    "   ENGINE = InnoDB"
    "   DEFAULT CHARACTER SET = utf8";
    ret = conn.query(sql);
    ASSERT(ret && ret->getRowCount() == 0);

    sql = " INSERT INTO `test` (pk, ftinyint, fvarchar) VALUES (1, 100, 'aa'), (2, 100, 'bb')";
    ret = conn.query(sql);
    ASSERT(ret && ret->getRowCount() == 0);

    sql = 
        "SELECT `pk`,`ftinyint`, `fsmallint`,`fint`,`fbigint`,`fchar`,`fvarchar`,`ffloat`,`fdouble`,"
        " `fdecimal`,`fdate`,`ftime`,`fdatetime`,`ftimestamp`,`fblob`,`fbinary`,`ftext` as xx "
        " FROM `test` where ftinyint=1";

    ret = conn.query(sql);
    for (size_t i = 0; i != ret->getRowCount(); ++i) {
        auto row = ret->getRow(i);
        ASSERT(row["ftinyint"]->getInt32() == 1);
        ASSERT(row["fvarchar"]->getString() == "aa");

        ASSERT(!row["pk"]->isNull());
        ASSERT(!row["ftinyint"]->isNull());
        ASSERT(!row["fvarchar"]->isNull());

        ASSERT(row["fsmallint"]->isNull());
        ASSERT(row["fint"]->isNull());
        ASSERT(row["fbigint"]->isNull());
        ASSERT(row["fchar"]->isNull());
        ASSERT(row["ffloat"]->isNull());
        ASSERT(row["fdouble"]->isNull());
        ASSERT(row["fdecimal"]->isNull());
        ASSERT(row["fdate"]->isNull());
        ASSERT(row["fdatetime"]->isNull());
        ASSERT(row["ftimestamp"]->isNull());
        ASSERT(row["fblob"]->isNull());
        ASSERT(row["fbinary"]->isNull());
        ASSERT(row["ftext"]->isNull());
        ASSERT(row["xx"]->isNull());        //别名
    }

    {
        //测试int类型
        sql = "DELETE FROM `test`";
        ret = conn.query(sql);
        ASSERT(ret && ret->getRowCount() == 0);

        sql = " INSERT INTO `test` (pk, ftinyint, fsmallint, fint, fbigint) VALUES "
            " (0, 100, 32767, 2147483647, 9223372036854775807)";
        ret = conn.query(sql);
        ASSERT(ret && ret->getRowCount() == 0);

        sql = "SELECT `ftinyint`, `fsmallint`,`fint`,`fbigint` FROM `test` where pk=0";
        ret = conn.query(sql);
        auto row = ret->getRow(0);
        ASSERT((int)row["ftinyint"]->getInt8() == 100);
        ASSERT(row["fsmallint"]->getInt16() == 32767);
        ASSERT(row["fint"]->getInt32() == 2147483647);
        ASSERT(row["fbigint"]->getInt64() == 9223372036854775807);
    }

    {
        //测试string float类型
        sql = "DELETE FROM `test`";
        ret = conn.query(sql);
        ASSERT(ret && ret->getRowCount() == 0);
        sql = " INSERT INTO `test` (pk, fchar, fvarchar, ffloat, fdouble) VALUES "
            " (0, 'aaa', 'bbb', 12345, 12345.6789)";
        ret = conn.query(sql);
        ASSERT(ret && ret->getRowCount() == 0);

        sql = "SELECT `fchar`, `fvarchar`,`ffloat`,`fdouble`  FROM `test` where pk=0";
        ret = conn.query(sql);
        auto row = ret->getRow(0);
        ASSERT(row["fchar"]->getString() == "aaa");
        ASSERT(row["fvarchar"]->getString() == "bbb");
        ASSERT(row["ffloat"]->getFloat() == 12345.f);
        ASSERT(row["fdouble"]->getDouble() == double(12345.6789));
    }

    {
        //测试datetime类型
        sql = "DELETE FROM `test`";
        ret = conn.query(sql);
        ASSERT(ret && ret->getRowCount() == 0);

        time_t tnow = 1475791730;   //2016-10-07 06:08:50
        sql = " INSERT INTO `test` (pk, fdate, ftime, fdatetime, ftimestamp) VALUES "
            " (0, ?, ?, ?, ?)";
        stmt = conn.prepareStatement(sql);
        ASSERT(stmt);
        stmt->setDateTime(0, mysqlcpp::DateTime(tnow));     //保存date类型
        stmt->setDateTime(1, mysqlcpp::DateTime(tnow));     //保存time类型
        stmt->setDateTime(2, mysqlcpp::DateTime(tnow));     //保存datetime类型
        stmt->setDateTime(3, mysqlcpp::DateTime());         //保存默认值
        auto ret_ex = conn.query(*stmt);
        ASSERT(ret_ex && ret_ex->getRowCount() == 0);

        sql = "SELECT `fdate`, `ftime`,`fdatetime`,`ftimestamp`  FROM `test` where pk=0";
        ret = conn.query(sql);
        auto row = ret->getRow(0);
        ASSERT(row["fdate"]->getDateTime().getString() == "2016-10-07");        //date类型
        ASSERT(row["ftime"]->getDateTime().getString() == "06:08:50");          //time类型
        ASSERT(row["fdatetime"]->getDateTime().getString() == "2016-10-07 06:08:50");   //datetime类型
        ASSERT(row["fdatetime"]->getDateTime().getTime() == tnow);                      //获取time_t
        ASSERT(row["ftimestamp"]->getDateTime().getString() == "0000-00-00 00:00:00");  //默认值
    }

    {
        //测试update
        sql = "DELETE FROM `test`";
        ret = conn.query(sql);
        ASSERT(ret && ret->getRowCount() == 0);
        sql = "INSERT INTO `test` (pk, fint, fvarchar) VALUES  (0, 100, 'xxx')";
        auto ret_ex = conn.query(*stmt);
        ASSERT(ret_ex && ret_ex->getRowCount() == 0);

        sql = "update `test` set `fint`=?, `fvarchar`=? where pk=?";
        stmt = conn.prepareStatement(sql);
        stmt->setInt32(0, 100);
        stmt->setString(1, "xxx");
        stmt->setInt32(2, 0);
        ret_ex = conn.query(*stmt);
        ASSERT(ret_ex && ret_ex->getRowCount() == 0);
        sql = "select * from test where pk=0";
        ret = conn.query(sql);
        auto row = ret->getRow(0);
        ASSERT(row["fint"]->getInt32() == 100);
        ASSERT(row["fvarchar"]->getString() == "xxx");

        stmt->clearParameters();
        stmt->setInt32(0, 200);
        stmt->setNull(1);
        stmt->setInt32(2, 0);
        ret_ex = conn.query(*stmt);
        ASSERT(ret_ex && ret_ex->getRowCount() == 0);
        sql = "select * from test where pk=0";
        ret = conn.query(sql);
        row = ret->getRow(0);
        ASSERT(row["fint"]->getInt32() == 200);
        ASSERT(row["fvarchar"]->isNull());
    }

    {
        //测试transaction
        mysqlcpp::Transaction transaction{conn};

        sql = "update `test` set `fint`=?, `fvarchar`=? where pk=?";
        stmt = conn.prepareStatement(sql);
        stmt->setInt32(0, 9900);
        stmt->setString(1, "hh");
        stmt->setInt32(2, 0);
        conn.query(*stmt);

        sql = "update `test` set `fit`=1, `fvarchar`=? where pk=?";
        auto ret = conn.query(sql);
        if (ret) {
            std::cout << "transaction commit\n";
            transaction.commit();
        } else {
            std::cout << "transaction rollback\n";
        }
    }

    return;
}


int main()
{
    mysqlcpp::initLog(&std::cout);
    test();

    return 0;
}
