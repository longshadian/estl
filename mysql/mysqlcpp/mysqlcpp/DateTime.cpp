#include "mysqlcpp/DateTime.h"

#include <ctime>
#include <time.h>
#include <cstring>
#include <array>

#include "mysqlcpp/Utility.h"

namespace mysqlcpp {

DateTime::DateTime()
    : m_mysql_time()
{
    std::memset(&m_mysql_time, 0, sizeof(m_mysql_time));
}

DateTime::DateTime(time_t t)
    : DateTime(boost::posix_time::from_time_t(t))
{
}

DateTime::DateTime(const timeval& tv)
    : DateTime(utility::timevalToPtime(tv))
{
}

DateTime::DateTime(const MYSQL_TIME& mysql_time)
    : m_mysql_time(mysql_time)
{
}

DateTime::DateTime(const boost::posix_time::ptime& pt)
    : DateTime()
{
    auto date = pt.date();
    m_mysql_time.year = date.year();
    m_mysql_time.month = date.month();
    m_mysql_time.day = date.day();

    auto td = pt.time_of_day();
    m_mysql_time.hour = unsigned int(td.hours());
    m_mysql_time.minute = unsigned int(td.minutes());
    m_mysql_time.second = unsigned int(td.seconds());
    m_mysql_time.second_part = unsigned int(td.fractional_seconds());
}

DateTime::DateTime(const DateTime& rhs)
    : m_mysql_time(rhs.m_mysql_time)
{
}

DateTime& DateTime::operator=(const DateTime& rhs)
{
    if (this != &rhs) {
        m_mysql_time = rhs.m_mysql_time;
    }
    return *this;
}

DateTime::DateTime(DateTime&& rhs)
    : m_mysql_time(std::move(rhs.m_mysql_time))
{
}

DateTime& DateTime::operator=(DateTime&& rhs)
{
    if (this != &rhs) {
        m_mysql_time = std::move(rhs.m_mysql_time);
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

const MYSQL_TIME& DateTime::getMYSQL_TIME() const
{
    return m_mysql_time;
}

MYSQL_TIME& DateTime::getMYSQL_TIME()
{
    return m_mysql_time;
}

std::string DateTime::getString() const
{
    std::array<char, 128> arr{};
    arr.fill(0);
    if (m_mysql_time.second_part == 0) {
        snprintf(arr.data(), arr.size(),"%04d-%02d-%02d %02d:%02d:%02d"
            , m_mysql_time.year, m_mysql_time.month, m_mysql_time.day
            , m_mysql_time.hour, m_mysql_time.minute, m_mysql_time.second
            );
    } else {
        snprintf(arr.data(), arr.size(),"%04d-%02d-%02d %02d:%02d:%02d.%06ld"
            , m_mysql_time.year, m_mysql_time.month, m_mysql_time.day
            , m_mysql_time.hour, m_mysql_time.minute, m_mysql_time.second
            , m_mysql_time.second_part
            );
    }
    return std::string{arr.data()};
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

