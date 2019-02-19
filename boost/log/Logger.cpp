#include "Logger.h"

#include <ios>

#include <boost/log/support/date_time.hpp> 
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>  

namespace logger {

void init(std::string path)
{
    LogOptional opt{};
    opt.m_file_name_pattern = std::move(path);
    init(opt);
}

void init(const LogOptional& opt)
{
    Logger::getInstance().init(opt);
}

void stop()
{
    Logger::getInstance().stop();
}

void logFormat(LOG_TYPE log_type, const char* format, ...)
{
    char buff[1024] = {0};
    va_list vl;
    va_start(vl, format);
    ::vsnprintf(buff, sizeof(buff), format, vl);
    va_end(vl);
    BOOST_LOG_SEV(logger::Logger::getInstance().m_lg, log_type) << buff;
}

Logger::Logger()
    : m_sink(nullptr)
    , m_lg()
{

}

Logger::~Logger()
{
    stop();
}

Logger& Logger::getInstance()
{
    static Logger instance_;
    return instance_;
}

void Logger::init(const LogOptional& opt)
{
    if (opt.m_file_name_pattern.empty())
        return;
    if (m_sink)
        return;

    boost::shared_ptr<boost::log::sinks::text_file_backend> backend = boost::make_shared<boost::log::sinks::text_file_backend>();
    backend->auto_flush(opt.m_auto_flush);
    backend->set_open_mode(opt.m_open_mode);
    backend->set_file_name_pattern(opt.m_file_name_pattern);
    backend->set_rotation_size(opt.m_rotation_size);
    //backend->set_time_based_rotation(boost::log::sinks::file::rotation_at_time_point(0, 0, 0));
    backend->set_time_based_rotation(opt.m_rotation_at_time_point);

    m_sink = boost::make_shared<FileSink>(backend);
    m_sink->set_formatter(
        boost::log::expressions::stream
        << boost::log::expressions::format_date_time< boost::posix_time::ptime >("TimeStamp", "%H:%M:%S.%f")
        << "\t" << boost::log::expressions::attr<LOG_TYPE>("Severity")
        << "\t" << boost::log::expressions::message
    );

    boost::log::core::get()->add_sink(m_sink);

    boost::log::add_common_attributes();
}

void Logger::stop()
{
    if (m_sink) {
        boost::shared_ptr<boost::log::core> core = boost::log::core::get();
        core->remove_sink(m_sink);
        m_sink->stop();
        m_sink->flush();
        m_sink.reset();
    }
}

}
