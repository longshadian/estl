
/*
#ifndef BOOST_SYSTEM_NO_DEPRECATED
#define BOOST_SYSTEM_NO_DEPRECATED 1
#endif
*/

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>

//#include <boost/date_time.hpp>
//#include <boost/date_time/posix_time/posix_time.hpp>

#include <boost/log/common.hpp>  
#include <boost/log/expressions.hpp>  
#include <boost/log/utility/setup/file.hpp>  
#include <boost/log/utility/setup/console.hpp>  
#include <boost/log/utility/setup/common_attributes.hpp>  
#include <boost/log/attributes/timer.hpp>  
#include <boost/log/attributes/named_scope.hpp>  
#include <boost/log/sources/logger.hpp>  
#include <boost/log/support/date_time.hpp> 




#include <boost/log/sinks/async_frontend.hpp>
// Related headers
#include <boost/log/sinks/unbounded_fifo_queue.hpp>
#include <boost/log/sinks/unbounded_ordering_queue.hpp>
#include <boost/log/sinks/bounded_fifo_queue.hpp>
#include <boost/log/sinks/bounded_ordering_queue.hpp>
#include <boost/log/sinks/drop_on_overflow.hpp>
#include <boost/log/sinks/block_on_overflow.hpp>


#include <thread>
#include <chrono>
#include <ios>

enum LOG_TYPE
{
    log_type_error,
    log_type_normal,
};

template< typename CharT, typename TraitsT >
inline std::basic_ostream< CharT, TraitsT >& operator<< (
    std::basic_ostream< CharT, TraitsT >& strm, LOG_TYPE lvl)
{
    static const char* const str[] =
    {
        "log_type_error",
        "log_type_normal",
    };
    if (static_cast<std::size_t>(lvl) < (sizeof(str) / sizeof(*str)))
        strm << str[lvl];
    else
        strm << static_cast<int>(lvl);
    return strm;
}

typedef boost::log::sinks::asynchronous_sink<boost::log::sinks::text_file_backend> file_sink;

void init()
{
    //typedef boost::log::sinks::synchronous_sink<boost::log::sinks::text_file_backend> file_sink;
    /*
    boost::shared_ptr<file_sink> sink(new file_sink(
        //boost::log::keywords::file_name = "./log/sign_%Y-%m-%d_%H-%M-%S.log",
        boost::log::keywords::file_name = "./log/sign_%Y-%m-%d_%H.%3N.log",
        boost::log::keywords::rotation_size = 1024 * 1024,
        boost::log::keywords::time_based_rotation = boost::log::sinks::file::rotation_at_time_point(18, 57, 0),
        //boost::log::keywords::auto_flush = true,
        //boost::log::keywords::open_mode = std::ios_base::app
        boost::log::keywords::format = boost::log::expressions::stream
            << boost::log::expressions::format_date_time< boost::posix_time::ptime >("TimeStamp", "%Y-%m-%d %H:%M:%S.%f")
            << "\t" << boost::log::expressions::attr<LOG_TYPE>("Severity")  
            << "\t" << boost::log::expressions::message
        ));
    */

    boost::shared_ptr<boost::log::sinks::text_file_backend> backend = boost::make_shared<boost::log::sinks::text_file_backend>();
    backend->auto_flush(true);
    backend->set_open_mode(std::ios::app);
    backend->set_file_name_pattern("./log/sign_%Y-%m-%d_%H.%3N.log");
    backend->set_rotation_size(1024);
    backend->set_time_based_rotation(boost::log::sinks::file::rotation_at_time_point(18, 57, 0));

    boost::shared_ptr<file_sink> sink(new file_sink(backend));
    sink->set_formatter(
        boost::log::expressions::stream
            << boost::log::expressions::format_date_time< boost::posix_time::ptime >("TimeStamp", "%Y-%m-%d %H:%M:%S.%f")
            << "\t" << boost::log::expressions::attr<LOG_TYPE>("Severity")  
            << "\t" << boost::log::expressions::message
    );

    boost::log::core::get()->add_sink(sink);

    /*
    boost::log::add_file_log(
        //boost::log::keywords::file_name = "./log/sign_%Y-%m-%d_%H-%M-%S.log",
        boost::log::keywords::file_name = "./log/sign_%Y-%m-%d_%H.%3N.log",
        boost::log::keywords::rotation_size = 1024 * 1024,
        boost::log::keywords::time_based_rotation = boost::log::sinks::file::rotation_at_time_point(18, 57, 0),
        //boost::log::keywords::severity = severity,
        //boost::log::keywords::format = ("[%TimeStamp%]: %Message%"),
        boost::log::keywords::auto_flush = true,
        boost::log::keywords::open_mode = std::ios_base::app,
        boost::log::keywords::format = boost::log::expressions::stream
            << boost::log::expressions::format_date_time< boost::posix_time::ptime >("TimeStamp", "%Y-%m-%d %H:%M:%S.%f")
            << "\t" << boost::log::expressions::attr<LOG_TYPE>("Severity")  
            << "\t" << boost::log::expressions::message
        );
    //boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
    */

}

void stop_logging(boost::shared_ptr<file_sink>& sink)
{
    boost::shared_ptr<boost::log::core> core = boost::log::core::get();

    // Remove the sink from the core, so that no records are passed to it
    core->remove_sink(sink);

    // Break the feeding loop
    sink->stop();

    // Flush all log records that may have left buffered
    sink->flush();

    sink.reset();
}


int main()
{
    init();
    boost::log::add_common_attributes();

    boost::log::sources::severity_logger<LOG_TYPE> lg;
    while (true) {
        BOOST_LOG_SEV(lg, log_type_error) << __LINE__ << "\t" << __FILE__ << "\t" << __FUNCTION__ << " log_type_error severity message";
        BOOST_LOG_SEV(lg, log_type_normal) << __LINE__ << "\t" << __FILE__ << "\t" << __FUNCTION__ << " log_type_normal severity message";

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}