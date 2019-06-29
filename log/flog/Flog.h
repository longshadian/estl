#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdarg>

#include <vector>
#include <list>
#include <thread>
#include <string>
#include <condition_variable>
#include <mutex>
#include <atomic>

namespace flog
{

struct LogBuffer
{
    LogBuffer();
    ~LogBuffer();
    LogBuffer(const LogBuffer&) = delete;
    LogBuffer& operator=(const LogBuffer&) = delete;
    LogBuffer(LogBuffer&&);
    LogBuffer& operator=(LogBuffer&&);

    void Resize(std::size_t length);
    std::uint8_t* WritePtr();
    const std::uint8_t* ReadPtr() const;
    void Consume(std::size_t length);

    std::vector<std::uint8_t> m_buffer;
};

enum LOG_TYPE
{
    LOG_DEBUG = 1,
    LOG_TIP = 2,
    LOG_TJ = 4,
    LOG_ERROR = 8,
    LOG_TRACE = 16,
};

class LogRecord
{
public:
    LogRecord(const char* log_root, const char* app_name);
    ~LogRecord();
    void log(const char* log_type, const char* log_content);

    char	        m_app_name[256];
    std::int32_t    m_day;
    std::int32_t    m_log_type;
    std::string     m_dir_name;
    LogBuffer       m_buffer;
};

using LogRecordUPtr = std::unique_ptr<LogRecord>;

class Logger
{
    Logger();
public:
    struct log_t
    {
        int  m_type;
        char m_app_name[64];
        char m_app_log[1024];
    };
public:
    ~Logger();
    static Logger* Get();

    void Init();
    void setLogLevel(int level);
    int getLogLevel() const;

    void Post(LogRecordUPtr record);

private:
    void StartThread();
    std::string getFullAppName();
    std::string getRootDir();
    static int getexepath(char* path, int max);
    void WriteRecord(LogRecordUPtr record);

private:
    std::int32_t            m_log_level;
    std::atomic<bool>       m_running;
    std::thread             m_thread;
    std::string             m_log_root;
    std::mutex                      m_mtx;
    std::condition_variable         m_cond;
    std::list<LogRecordUPtr>        m_list;
};

void PrintFormat(const std::string& sub_dir_name, LOG_TYPE log_type, const char* format, ...);
void StringPrintf(std::vector<char>* output, const char* format, va_list args);

std::unique_ptr<LogRecord> CreateRecord();

} // namespace flog
