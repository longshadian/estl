#pragma once

#include <mysql.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <boost/date_time.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/date_time/c_local_time_adjustor.hpp>

#include "mysqlcpp/Types.h"

namespace mysqlcpp {

class DateTime;

namespace utility {

bool MYSQLCPP_EXPORT stringTo_Date(const std::string& str, unsigned int* year, unsigned int* month, unsigned int* day);
bool MYSQLCPP_EXPORT stringTo_DateTime_Timestamp(const std::string& str
    , unsigned int* year, unsigned int* month, unsigned int* day
    , unsigned int* hour, unsigned int* minute, unsigned int* second
    , unsigned long* microsecond);

boost::posix_time::ptime timevalToPtime(const struct timeval& tv);

void MYSQLCPP_EXPORT bindFiledsMeta(MYSQL_RES* mysql_res, std::vector<FieldMeta>* fields_data);

template <typename T>
void bzero(T* t)
{
    static_assert(std::is_pod<T>::value, "T must be POD!");
    std::memset(t, 0, sizeof(T));
}

/*
inline
boost::posix_time::ptime utcToLocal(time_t t)
{
    return boost::date_time::c_local_adjustor<boost::posix_time::ptime>::utc_to_local(
        boost::posix_time::from_time_t(t));
}

inline
struct tm time_t_TO_tm(time_t t)
{
    return boost::posix_time::to_tm(utcToLocal(t));
}
*/


} // utility
} // mysqlcpp
