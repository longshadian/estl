#pragma once

#include <mysql.h>
#include <vector>

#include "mysqlcpp/Types.h"
#include "mysqlcpp/DateTime.h"

namespace mysqlcpp {

class SQLString;
class FieldMeta;
class Connection;
class Field;

class MYSQLCPP_EXPORT Statement
{
public:
    Statement(Connection& conn);
    ~Statement();
    Statement(const Statement& right) = delete;
    Statement& operator=(const Statement& right) = delete;
public:
    Connection* getConnection();

    bool            execute(const std::string& sql);
    ResultSetPtr    executeQuery(const std::string& sql);

private:
    bool        checkConnection();
    ResultSetPtr createResultSet();
    bool storeResult(MYSQL_RES* mysql_res, std::vector<RowData>* rows, const FieldData* fields_data);

private:
    Connection& m_conn;
};

}
