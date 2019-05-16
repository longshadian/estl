#include "DateTime.h"

#include <ctime>
#include <time.h>
#include <cstring>

#include "Utils.h"

namespace mysqlcpp {

DateTime::DateTime()
    : m_mysql_time()
    , m_out_str()
{
    std::memset(&m_mysql_time, 0, sizeof(m_mysql_time));
}

DateTime::DateTime(time_t t)
    : DateTime()
{
    setMysqlTime(getLocaltime(t));
}

DateTime::DateTime(const timeval& t)
    :DateTime()
{
    setMysqlTime(getLocaltime(t.tv_sec));
    m_mysql_time.second_part = t.tv_usec;
}

DateTime::DateTime(const std::string& str)
    : DateTime()
{
    //0000-00-00 00:00:00
    if (str.length() == 19)
        datetimeFromString(str);
    else if (str.length() == 10)
        dateFromString(str);
    else if (str.length() == 8)
        timeFromString(str);
}

DateTime::DateTime(const char* str)
    :DateTime(std::string(str))
{
}

DateTime::DateTime(const std::string& str, enum_field_types type)
    :DateTime()
{
    m_out_str = str;
    switch (type)
    {
    case MYSQL_TYPE_TIMESTAMP:
        datetimeFromString(str);
        break;
    case MYSQL_TYPE_DATE:
        dateFromString(str);
        break;
    case MYSQL_TYPE_TIME:
        timeFromString(str);
        break;
    case MYSQL_TYPE_DATETIME:
        datetimeFromString(str);
        break;
    case MYSQL_TYPE_TIMESTAMP2:
        break;
    case MYSQL_TYPE_DATETIME2:
        break;
    case MYSQL_TYPE_TIME2:
        break;
    default:
        break;
    }
}

DateTime::DateTime(const DateTime& rhs)
    : m_mysql_time(rhs.m_mysql_time)
    , m_out_str(rhs.m_out_str)
{
}

DateTime& DateTime::operator=(const DateTime& rhs)
{
    if (this != &rhs) {
        m_mysql_time = rhs.m_mysql_time;
        m_out_str = rhs.m_out_str;
    }
    return *this;
}

DateTime::DateTime(DateTime&& rhs)
    : m_mysql_time(std::move(rhs.m_mysql_time))
    , m_out_str(std::move(rhs.m_out_str))
{
}

DateTime& DateTime::operator=(DateTime& rhs)
{
    if (this != &rhs) {
        m_mysql_time = std::move(rhs.m_mysql_time);
        m_out_str = std::move(rhs.m_out_str);
    }
    return *this;
}


DateTime::~DateTime()
{

}

std::vector<uint8> DateTime::getBinary() const
{
    std::vector<uint8> buffer{};
    buffer.resize(sizeof(m_mysql_time));
    std::memcpy(buffer.data(), &m_mysql_time, buffer.size());
    return buffer;
}

std::string DateTime::getString() const
{
    return m_out_str;
}

time_t DateTime::getTime() const
{
    if (isNull())
        return 0;
    struct tm tms{};
    std::memset(&tms, 0, sizeof(tms));
    tms.tm_year = m_mysql_time.year - 1900;
    tms.tm_mon = m_mysql_time.month - 1;
    tms.tm_mday = m_mysql_time.day;
    tms.tm_hour = m_mysql_time.hour;
    tms.tm_min = m_mysql_time.minute;
    tms.tm_sec = m_mysql_time.second;
    return std::mktime(&tms);
}

void DateTime::setMysqlTime(const std::array<unsigned long, 6>& arr)
{
    m_mysql_time.year = arr[0];
    m_mysql_time.month = arr[1];
    m_mysql_time.day = arr[2];
    m_mysql_time.hour = arr[3];
    m_mysql_time.minute = arr[4];
    m_mysql_time.second = arr[5];
    m_mysql_time.second_part = 0;
    m_mysql_time.neg = 0;
    m_mysql_time.time_type = MYSQL_TIMESTAMP_DATETIME;
}

std::array<unsigned long, 6> DateTime::getLocaltime(time_t t)
{
    std::array<unsigned long, 6> val{0};
    struct tm tms;
    struct tm* ptm = localtime_r(&t, &tms);
    val[0] = ptm->tm_year + 1900;
    val[1] = ptm->tm_mon + 1;
    val[2] = ptm->tm_mday;
    val[3] = ptm->tm_hour;
    val[4] = ptm->tm_min;
    val[5] = ptm->tm_sec;
    return val;
}

void DateTime::datetimeFromString(const std::string& str)
{
    util::Tokenizer tk{str, ' '};
    if (tk.size() != 2)
        return;
    dateFromString(tk[0]);
    timeFromString(tk[1]);
}

void DateTime::dateFromString(const std::string& str)
{
    util::Tokenizer tk{str, '-'};
    if (tk.size() != 3)
        return;
    m_mysql_time.year = std::strtol(tk[0], nullptr, 10);
    m_mysql_time.month = std::strtol(tk[1], nullptr, 10);
    m_mysql_time.day = std::strtol(tk[2], nullptr, 10);
}

void DateTime::timeFromString(const std::string& str)
{
    util::Tokenizer tk{str, ':'};
    if (tk.size() != 3)
        return;
    m_mysql_time.hour = std::strtol(tk[0], nullptr, 10);
    m_mysql_time.minute = std::strtol(tk[1], nullptr, 10);
    m_mysql_time.second = std::strtol(tk[2], nullptr, 10);
}

bool DateTime::isNull() const
{
    return m_mysql_time.year == 0 
        && m_mysql_time.month == 0
        && m_mysql_time.day == 0
        && m_mysql_time.hour == 0
        && m_mysql_time.minute == 0
        && m_mysql_time.second == 0
        && m_mysql_time.second_part == 0;
}

}

