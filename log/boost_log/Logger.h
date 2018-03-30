#pragma once

#include <string>

#include <boost/log/trivial.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/sinks/async_frontend.hpp>

#define LOGGER_TRACE(log_type) \
    BOOST_LOG_SEV(logger::Logger::getInstance().m_lg, log_type)

namespace logger {

typedef boost::log::sinks::asynchronous_sink<boost::log::sinks::text_file_backend> FileSink;

enum LOG_TYPE
{
    ERROR,
    NORMAL,
};

template< typename CharT, typename TraitsT >
inline std::basic_ostream< CharT, TraitsT >& operator<< (
    std::basic_ostream< CharT, TraitsT >& strm, LOG_TYPE lvl)
{
    static const char* const str[] = { "ERROR", "NORMAL", };
    if (static_cast<std::size_t>(lvl) < (sizeof(str) / sizeof(*str)))
        strm << str[lvl];
    else
        strm << static_cast<int>(lvl);
    return strm;
}

void init(std::string path);

class Logger
{
    Logger();
public:
    ~Logger();
    static Logger& getInstance();
    void init(std::string file_name_pattern);
private:
    boost::shared_ptr<FileSink>  m_sink;
public:
    boost::log::sources::severity_logger<LOG_TYPE> m_lg;
};


}

