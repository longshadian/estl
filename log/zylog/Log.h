#ifndef _ZYLIB_LOG_H_
#define _ZYLIB_LOG_H_

#include <cstdarg>
#include <atomic>
#include <string>
#include <memory>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <list>
#include <sstream>

namespace zylib {

enum LOG_SEVERITY
{
    DEBUG    = 0,
    TIP      = 1,
    WARNING  = 2,
    ERROR    = 3,
    FATAL    = 4,
};

const char* getLogSeverityName(LOG_SEVERITY s);

struct LogStream
{
    std::string getString();
    std::ostringstream m_ostm;
};


struct Logger
{
    Logger(LOG_SEVERITY s);
    ~Logger();

    LogStream& stream();
    LOG_SEVERITY    m_severity;
    LogStream       m_stream;
};

class ServerLogger
{
    class LogFile
    {
    public:
        LogFile(std::string path, std::string file_name);
        ~LogFile();
        void flush(const char* severity_name, const char* log_content, size_t length);
    private:
        FILE*       m_file;
        int	        m_day;
        std::string	m_full_file_name;
    };

    ServerLogger();
public:
    enum {MIN_LOG_MSG_LENGTH = 128};
    enum {MAX_LOG_MSG_LENGTH = 1024};

    struct LogMessage
    {
        LOG_SEVERITY        m_severity;
        std::vector<char>   m_buffer;
        std::string         m_content;
    };
public:
    ~ServerLogger();
    static ServerLogger& getInstance();

    static void init(std::string path, std::string file_name, LOG_SEVERITY s = DEBUG);
    static void printLogFormat(LOG_SEVERITY log_type, const char* format, ...);
    static void printLog(LOG_SEVERITY log_type, std::string s);

    LOG_SEVERITY getLogSeverity() const;
private:
    void start(std::string path_end_with_sprit, std::string file_name, LOG_SEVERITY s);
    void run();
    void pushMsg(std::unique_ptr<LogMessage> msg);

    static struct tm* localtimeEx(const time_t* t, struct tm* output);
    static void stringPrintf(std::vector<char>* output, const char* format, va_list args);
private:
    std::string                             m_path;
    std::string                             m_file_name;
    LOG_SEVERITY                            m_log_severity;
    std::shared_ptr<LogFile>                m_log_file;

    std::atomic<bool>                       m_running;
    std::thread                             m_thread;
    std::mutex                              m_mtx;
    std::condition_variable                 m_cond;
    std::list<std::unique_ptr<LogMessage>>  m_log_messages;
};

}

/*
zylib::LogStream& operator<<(zylib::LogStream& os, bool v);
zylib::LogStream& operator<<(zylib::LogStream& os, char v);
zylib::LogStream& operator<<(zylib::LogStream& os, signed char v);
zylib::LogStream& operator<<(zylib::LogStream& os, unsigned char v);
zylib::LogStream& operator<<(zylib::LogStream& os, short v);
zylib::LogStream& operator<<(zylib::LogStream& os, unsigned short v);
zylib::LogStream& operator<<(zylib::LogStream& os, int v);
zylib::LogStream& operator<<(zylib::LogStream& os, unsigned int v);
zylib::LogStream& operator<<(zylib::LogStream& os, long v);
zylib::LogStream& operator<<(zylib::LogStream& os, unsigned long v);
zylib::LogStream& operator<<(zylib::LogStream& os, long long v);
zylib::LogStream& operator<<(zylib::LogStream& os, unsigned long long v);
zylib::LogStream& operator<<(zylib::LogStream& os, float v);
zylib::LogStream& operator<<(zylib::LogStream& os, double v);
zylib::LogStream& operator<<(zylib::LogStream& os, long double v);
zylib::LogStream& operator<<(zylib::LogStream& os, std::string v);
zylib::LogStream& operator<<(zylib::LogStream& os, const char* v);
*/

template <typename T>
inline
zylib::LogStream& operator<<(zylib::LogStream& os, const T& v)
{
    os.m_ostm << v;
    return os;
}

#define PLOG(level) zylib::Logger(zylib::level).stream()

#endif
