#ifndef _MYSQLCPP_DATETIME_H
#define _MYSQLCPP_DATETIME_H

#include <mysql.h>
#include <array>
#include <vector>
#include <string>

#include "Types.h"

namespace mysqlcpp {

class DateTime
{
public:
    DateTime();
    explicit DateTime(time_t t);
    explicit DateTime(const timeval& t);
    explicit DateTime(const std::string& str);
    explicit DateTime(const char* str);
    DateTime(const std::string& str, enum_field_types type);

    DateTime(const DateTime& rhs);
    DateTime& operator=(const DateTime& rhs);

    DateTime(DateTime&& rhs);
    DateTime& operator=(DateTime& rhs);

    ~DateTime();
public:
    std::vector<uint8> getBinary() const;
    std::string        getString() const;
    time_t             getTime() const;
private:
    static std::array<unsigned long, 6> getLocaltime(time_t t);

    void setMysqlTime(const std::array<unsigned long, 6>& arr);
    void datetimeFromString(const std::string& str);
    void dateFromString(const std::string& str);
    void timeFromString(const std::string& str);
    bool isNull() const;
private:
    MYSQL_TIME m_mysql_time;
    std::string m_out_str;     
};

}

#endif
