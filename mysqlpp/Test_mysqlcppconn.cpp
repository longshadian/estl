#include <mysql++.h>

#include <string>

int main()
{
    mysqlpp::Connection conn{ false };
    mysqlpp::SQLQueryParms params;

    std::string s = "123456";

    mysqlpp::sql_blob val{s.data(), s.length()};
    params << val;

    mysqlpp::StoreQueryResult ret{};
    const mysqlpp::Row& row = ret[0];
    row["a"].data();
    row["a"].length();

    return 0;
}