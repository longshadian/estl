#include "zylib/Logger.h"

#include <ios>

#include <boost/core/null_deleter.hpp>
#include <boost/log/support/date_time.hpp> 
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>  
#include <boost/log/utility/exception_handler.hpp>


namespace zylib {

namespace logger {

BOOST_LOG_ATTRIBUTE_KEYWORD(severity, "Severity", SEVERITY)

void initAsyncFile(const FileOptional& opt, SEVERITY s)
{
    Logger::instance().initAsyncFile(opt, s);
}

void initSyncFile(const FileOptional& opt, SEVERITY s)
{
    Logger::instance().initSyncFile(opt, s);
}

void initSyncConsole(SEVERITY s)
{
    Logger::instance().initSyncConsole(s);
}

SafeExit::~SafeExit()
{
    Logger::instance().stop();
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

Logger::Logger()
    : m_init()
    , m_s()
{

}

Logger::~Logger()
{
    stop();
}

Logger& Logger::instance()
{
    static Logger instance_;
    return instance_;
}

void Logger::stop()
{
    if (!m_init.exchange(false))
        return;
    boost::shared_ptr<boost::log::core> core = boost::log::core::get();
    core->remove_all_sinks();
    /*
    core->remove_sink(m_sink);
    m_sink->stop();
    m_sink->flush();
    m_sink.reset();
    */
}

void Logger::initAsyncFile(const FileOptional& opt, SEVERITY s)
{
    if (opt.m_file_name_pattern.empty())
        return;
    if (m_init.exchange(true))
        return;

    auto backend = createTextFileBackend(opt);
    auto frontend = boost::make_shared<AsyncTextFile>(backend);
    frontend->set_formatter( creatLogFormattter() );
    frontend->set_filter(severity >= s);
    frontend->set_exception_handler(boost::log::make_exception_suppressor());
    boost::log::core::get()->add_sink(frontend);
    boost::log::core::get()->set_exception_handler(boost::log::make_exception_suppressor());
    boost::log::add_common_attributes();
}

// 同步输出至文件
void Logger::initSyncFile(const FileOptional& opt, SEVERITY s)
{
    if (opt.m_file_name_pattern.empty())
        return;
    if (m_init.exchange(true))
        return;

    auto backend = createTextFileBackend(opt);
    auto frontend = boost::make_shared<SyncTextFile>(backend);
    frontend->set_formatter( creatLogFormattter() );
    frontend->set_filter(severity >= s);
    frontend->set_exception_handler(boost::log::make_exception_suppressor());
    boost::log::core::get()->add_sink(frontend);
    boost::log::core::get()->set_exception_handler(boost::log::make_exception_suppressor());
    boost::log::add_common_attributes();
}

// 同步输出至控制台
void Logger::initSyncConsole(SEVERITY s)
{
    if (m_init.exchange(true))
        return;
    auto backend = boost::make_shared<TextOStreamBackend>();
    backend->add_stream(boost::shared_ptr<std::ostream>(&std::cout, boost::null_deleter()));
    auto frontend = boost::make_shared<SyncConsole>(backend);
    frontend->set_formatter(creatLogFormattter());
    frontend->set_filter(severity >= s);
    frontend->set_exception_handler(boost::log::make_exception_suppressor());
    boost::log::core::get()->add_sink(frontend);
    boost::log::core::get()->set_exception_handler(boost::log::make_exception_suppressor());
    boost::log::add_common_attributes();
}

boost::shared_ptr<TextFileBackend> Logger::createTextFileBackend(const FileOptional& opt)
{
    auto backend = boost::make_shared<TextFileBackend>();
    backend->auto_flush(opt.m_auto_flush);
    backend->set_open_mode(opt.m_open_mode);
    backend->set_file_name_pattern(opt.m_file_name_pattern);
    backend->set_rotation_size(opt.m_rotation_size);
    backend->set_time_based_rotation(opt.m_rotation_at_time_point);
    return backend;
}

boost::log::formatter Logger::creatLogFormattter()
{
    return
        boost::log::expressions::stream
        << '['
        << boost::log::expressions::format_date_time< boost::posix_time::ptime >("TimeStamp", "%H:%M:%S.%f")
        << ' ' << boost::log::expressions::attr<SEVERITY>("Severity")
        << ' ' << boost::log::expressions::message;
}

} // logger
} // zylib
