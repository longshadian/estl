#include "MysqlConnectPool.h"

#include <memory>
#include <chrono>
#include <mysql++/ssqls.h>
#include <iostream>
#include <array>
#include <fstream>

std::shared_ptr<MysqlConnectPool> db_pool = nullptr;
std::string db_name = "mytest";
std::string db_ip   = "192.168.125.130";
std::string db_user = "root";
std::string db_passwd = "123456";
int db_port         = 3306;
std::string db_charset = "utf8";


struct TimerTick
{
	TimerTick()
	{
		m_tp = std::chrono::system_clock::now();
	}

	~TimerTick()
	{
		auto tend = std::chrono::system_clock::now();
		auto milli_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(tend - m_tp).count();
		printf("%d.%d\n", (int)milli_seconds /1000, (int)milli_seconds %1000);
	}

	std::chrono::system_clock::time_point m_tp;
};

bool selectDB()
{
    try {
        mysqlpp::ScopedConnection conn(*db_pool);
        auto query = conn->query();
        query << "select ftime from ttest where fid = 1";
        mysqlpp::StoreQueryResult ret = query.store();
        for (size_t i = 0; i != ret.num_rows(); ++i) {
            const mysqlpp::Row & row = ret[i];
            std::cout << row["ftime"].c_str() << "\n";
            auto dt = mysqlpp::DateTime(row["ftime"]);
            std::cout << dt << "\n";
            std::cout << time_t(mysqlpp::DateTime(row["ftime"])) << "\n";
        }
        return true;
    } catch(const mysqlpp::Exception& e) {
        std::cout << "error:" << e.what() << "\n";
        return false;
    }
}

bool insertCards(int type)
{
    try {
        mysqlpp::ScopedConnection conn(*db_pool);
        auto query = conn->query();
        query.reset();
        query << " INSERT INTO tcard_policy (ftype,fbottom,fcard0,fcard1,fcard2) VALUES (%0,%1q,%2q,%3q,%4q)";
        query.parse();

        mysqlpp::SQLQueryParms params;
        params << type;
        query.execute(params);
        return true;
    } catch(const mysqlpp::Exception& e) {
        std::cout << "error:" << e.what() << "\n";
        return false;
    }
}

bool updateCards()
{
    const char* sql = "update ttest set ftime=%0q where fid = %1";
    try {
        mysqlpp::ScopedConnection conn(*db_pool);
        auto query = conn->query(sql);
        query.parse();
        mysqlpp::SQLQueryParms params;
        params << "";
        params << 1;
        query.execute(params);
        return true;
    } catch (const mysqlpp::Exception& e) {
        std::cout << "error:" << e.what() << "\n";
        return false;
    }
}

bool insertGBK()
{
    try {
        mysqlpp::ScopedConnection conn(*db_pool);
        auto query = conn->query();
        query.reset();
        query << "update `ttest` set fname = 'บวบว' where fid = 1";
        query.execute();
        return true;
    } catch(const mysqlpp::Exception& e) {
        std::cout << "error:" << e.what() << "\n";
        return false;
    }
}


bool selectBigData()
{
	int n = 0;
    const char* sql = " SELECT `fid`, `fname`, `fdesc`, `ftime`, `fsid`, `fdate` FROM `ttest_ex`";
    try {
        mysqlpp::ScopedConnection conn(*db_pool);
        auto query = conn->query(sql);
		auto ret = query.store();
		for (size_t i = 0; i != ret.num_rows(); ++i) {
			++n;
		}
        return true;
    } catch(const mysqlpp::Exception& e) {
        std::cout << "error:" << e.what() << "\n";
        return false;
    }
}

bool selectBigDataEx()
{
	int min_fid = 0;
	int max_fid = 0;
	{
		mysqlpp::ScopedConnection conn(*db_pool);
		auto query = conn->query();
		query << "select min(fid) as min_val from ttest_ex;";
		auto ret = query.store();
		min_fid = (int)ret[0]["min_val"];
	}

	{
		mysqlpp::ScopedConnection conn(*db_pool);
		auto query = conn->query();
		query << "select max(fid) as max_val from ttest_ex;";
		auto ret = query.store();
		max_fid = (int)ret[0]["max_val"];
	}


	//max_fid = 12;

    const int LIMIT = 10000;
	int n = min_fid;
	std::array<char, 1024> buffer{0};
	const char* sql = "select fid from ttest_ex where fid >= %d limit %d";
	try {
		int fid = 0;
		mysqlpp::ScopedConnection conn(*db_pool);
		auto query = conn->query();
		while (true) {
			buffer.fill(0);
			snprintf(buffer.data(), buffer.size(), sql, n, LIMIT);
			query.reset();
			query << buffer.data();
			query.parse();
			//std::cout << buffer.data() << "\n";
			auto ret = query.store();
			for (size_t i = 0; i != ret.num_rows(); ++i) {
				fid = (int)ret[i]["fid"];
				//std::cout << " " << fid;
			}
			//std::cout << "\n";

            n += LIMIT;
			if (max_fid < n)
				break;
		}
        std::cout << fid << "\n";
		return true;
	}
	catch (const mysqlpp::Exception& e) {
		std::cout << "error:" << e.what() << "\n";
		return false;
	}
}

int main()
{
    db_pool = std::make_shared<MysqlConnectPool>(db_name, db_ip, db_user, db_passwd, db_port, db_charset);
    //updateCards();
    //selectDB();
    //insertGBK();

	/*
	{
		TimerTick t{};
		selectBigData();
	}
	*/

	{
		TimerTick t{};
		selectBigDataEx();
	}
    return 0;
}
