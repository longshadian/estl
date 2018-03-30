#include "Logger.h"

#include <ios>

#include <boost/log/support/date_time.hpp> 
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>  

namespace logger {

void init(std::string path)
{
    Logger::getInstance().init(std::move(path));
}

Logger::Logger()
    : m_sink(nullptr)
    , m_lg()
{

}

Logger::~Logger()
{
    if (m_sink) {
        boost::shared_ptr<boost::log::core> core = boost::log::core::get();
        core->remove_sink(m_sink);
        m_sink->stop();
        m_sink->flush();
        m_sink.reset();
    }
}

Logger& Logger::getInstance()
{
    static Logger instance_;
    return instance_;
}

void Logger::init(std::string file_pattern)
{
    if (file_pattern.empty())
        return;
    if (m_sink)
        return;

    boost::shared_ptr<boost::log::sinks::text_file_backend> backend = boost::make_shared<boost::log::sinks::text_file_backend>();
    backend->auto_flush(true);
    backend->set_open_mode(std::ios::app);
    //backend->set_file_name_pattern("./log/sign_%Y-%m-%d_%H.%3N.log");
    backend->set_file_name_pattern(file_pattern);
    backend->set_rotation_size(1024 * 1024 * 10);
    backend->set_time_based_rotation(boost::log::sinks::file::rotation_at_time_point(0, 0, 0));

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

}
