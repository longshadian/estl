#pragma once

#include <string>

#include <boost/log/trivial.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/sinks/async_frontend.hpp>
#include <boost/log/sinks/sync_frontend.hpp>

#include "zylib/ZylibDefine.h"

//#define BOOST_LOG_DYN_LINK 1

#define LOG(S) \
    BOOST_LOG_SEV(zylib::logger::Logger::instance().m_s, zylib::logger::SEVERITY::S) \
        << __FILE__ << ':' << __LINE__ << ':' << __FUNCTION__ << "] "

namespace zylib {
namespace logger {

enum SEVERITY
{
    Debug,
    Info,
    Warning,
    Error,
};

template< typename CharT, typename TraitsT >
inline std::basic_ostream< CharT, TraitsT >& operator<< (
    std::basic_ostream< CharT, TraitsT >& strm, SEVERITY lvl)
{
    static const char* const str[] = {
        "Debug  ",
        "Info   ",
        "Warning",
        "Error  ",
    };
    if (static_cast<std::size_t>(lvl) < (sizeof(str) / sizeof(*str)))
        strm << str[static_cast<size_t>(lvl)];
    else
        strm << static_cast<int>(lvl);
    return strm;
}

struct FileOptional
{
    bool                    m_auto_flush{true};
    std::ios_base::openmode m_open_mode{std::ios::app};
    //std::string             m_file_name_pattern{"./log_%Y-%m-%d_%H.%3N.log"};
    std::string             m_file_name_pattern{"./log_%Y-%m-%d_%H.log"};
    uint32_t                m_rotation_size{1024 * 1024 * 100};
    boost::log::sinks::file::rotation_at_time_point m_rotation_at_time_point{0, 0, 0};
};

using TextOStreamBackend = boost::log::sinks::text_ostream_backend;
using TextFileBackend = boost::log::sinks::text_file_backend;

using AsyncTextFile = boost::log::sinks::asynchronous_sink<TextFileBackend>;
using SyncTextFile = boost::log::sinks::synchronous_sink<TextFileBackend>;
using SyncConsole = boost::log::sinks::synchronous_sink<TextOStreamBackend>;

// 异步输出至文件
void ZYLIB_EXPORT initAsyncFile(const FileOptional& opt, SEVERITY = SEVERITY::Debug);

// 同步输出至文件
void ZYLIB_EXPORT initSyncFile(const FileOptional& opt, SEVERITY = SEVERITY::Debug);

// 同步输出至控制台
void ZYLIB_EXPORT initSyncConsole(SEVERITY = SEVERITY::Debug);

struct ZYLIB_EXPORT SafeExit
{
    SafeExit() = default;
    ~SafeExit();
    SafeExit(const SafeExit& rhs) = delete;
    SafeExit& operator=(const SafeExit& rhs) = delete; 
};

class ZYLIB_EXPORT Logger
{
    Logger();
public:
    ~Logger();
    static Logger& instance();
    void stop();

    // 异步输出至文件
    void initAsyncFile(const FileOptional& opt, SEVERITY s);

    // 同步输出至文件
    void initSyncFile(const FileOptional& opt, SEVERITY s);

    // 同步输出至控制台
    void initSyncConsole(SEVERITY s);

private:
    static boost::shared_ptr<TextFileBackend> createTextFileBackend(const FileOptional& opt);
    static boost::log::formatter creatLogFormattter();

private:
    std::atomic<bool> m_init;
public:
    boost::log::sources::severity_logger<SEVERITY> m_s;
};


} /// logger

} /// zylib
