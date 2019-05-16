#include "mysqlcpp/Utility.h"

#include <regex>
#include <cstring>
#include <algorithm>

#include "mysqlcpp/DateTime.h"
#include "mysqlcpp/FieldMeta.h"
#include "mysqlcpp/MysqlcppLog.h"
#include "mysqlcpp/detail/Convert.h"

namespace mysqlcpp {

namespace utility {

bool stringTo_Date(const std::string& str, unsigned int* year, unsigned int* month, unsigned int* day)
{
    //YYYY-MM-DD
    if (str.size() != 10)
        return false;
    std::string pattern_str = R"((\d+)-(\d+)-(\d+))";
    try {
        std::regex pattern(pattern_str);
        std::smatch results{};
        if (std::regex_search(str, results, pattern)) {
            if (results.size() != 4)
                return false;
            *year = detail::Convert<unsigned int>::cvt_noexcept(results[1]);
            *month = detail::Convert<unsigned int>::cvt_noexcept(results[2]);
            *day = detail::Convert<unsigned int>::cvt_noexcept(results[3]);
            return true;
        }
        return false;
    } catch (const std::regex_error& e) {
        MYSQLCPP_LOG(Warning, "regex exception %s", e.what());
        return false;
    }
}

bool stringTo_DateTime_Timestamp(const std::string& str
    , unsigned int* year, unsigned int* month, unsigned int* day
    , unsigned int* hour, unsigned int* minute, unsigned int* second
    , unsigned long* second_part)
{
    if (str.empty())
        return false;
    try 
    {
        {
            //YYYY-MM-DD HH:MM:SS
            std::string pattern_str = R"((\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+))";
            std::regex pattern(pattern_str);
            std::smatch results{};
            if (std::regex_search(str, results, pattern)) {
                if (results.size() == 7) {
                    *year = detail::Convert<unsigned int>::cvt_noexcept(results[1]);
                    *month = detail::Convert<unsigned int>::cvt_noexcept(results[2]);
                    *day = detail::Convert<unsigned int>::cvt_noexcept(results[3]);
                    *hour = detail::Convert<unsigned int>::cvt_noexcept(results[4]);
                    *minute = detail::Convert<unsigned int>::cvt_noexcept(results[5]);
                    *second = detail::Convert<unsigned int>::cvt_noexcept(results[6]);
                    *second_part = 0;
                    return true;
                }
            }
        }

        {
            //YYYY-MM-DD HH:MM:SS.second_part
            std::string pattern_str = R"((\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+)\.(\d+))";
            std::regex pattern(pattern_str);
            std::smatch results{};
            if (std::regex_search(str, results, pattern)) {
                if (results.size() == 8) {
                    *year = detail::Convert<unsigned int>::cvt_noexcept(results[1]);
                    *month = detail::Convert<unsigned int>::cvt_noexcept(results[2]);
                    *day = detail::Convert<unsigned int>::cvt_noexcept(results[3]);
                    *hour = detail::Convert<unsigned int>::cvt_noexcept(results[4]);
                    *minute = detail::Convert<unsigned int>::cvt_noexcept(results[5]);
                    *second = detail::Convert<unsigned int>::cvt_noexcept(results[6]);
                    *second_part = detail::Convert<unsigned int>::cvt_noexcept(results[7]);
                    return true;
                }
            }
        }
        return false;
    } catch (const std::regex_error& e) {
        MYSQLCPP_LOG(Warning, "regex exception %s", e.what());
        return false;
    }
}

void bindFiledsMeta(MYSQL_RES* mysql_res, std::vector<FieldMeta>* fields_data)
{
    unsigned int field_count = ::mysql_num_fields(mysql_res);
    MYSQL_FIELD* mysql_fields = ::mysql_fetch_fields(mysql_res);
    fields_data->reserve(field_count);
    for (unsigned int i = 0; i != field_count; ++i) {
        //保存列元数据
        fields_data->push_back(FieldMeta(&mysql_fields[i], i));
    }
}

boost::posix_time::ptime timevalToPtime(const struct timeval& tv)
{
    auto pt = boost::posix_time::from_time_t(tv.tv_sec);
    pt += boost::posix_time::time_duration{0,0,0, tv.tv_usec};
    return pt;
}

} // utility
} // mysqlcpp
