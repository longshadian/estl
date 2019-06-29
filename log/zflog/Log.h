#pragma once

#include <cstdarg>
#include <cstdint>

#include <atomic>
#include <string>
#include <memory>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <list>

namespace zylib 
{

enum LOG_TYPE
{
	LOG_DEBUG       = 1,
	LOG_INFO        = 2,
	LOG_WARNING     = 4,
	LOG_ERROR       = 8
};

struct Record
{
    int m_type;
    std::string m_dir_name;
    std::vector<char> m_buffer;
};

class Sink
{
public:
    Sink(std::string log_root, std::string dir_name);
    ~Sink();
    void flush(const char* log_type, const char* log_content, size_t length);

    FILE*       m_file;
    int	        m_day;
    std::string	m_dir_name;
};

class Logger
{
    Logger();
public:
    ~Logger();
    static Logger& Get();

    static void init(std::string path);

    void setLogLevel(int level);
    int getLogLevel() const;
private:
    void start(std::string path_end_with_sprit);
    void StartThread();
    void pushMsg(std::unique_ptr<Record> msg);

private:
    int                     m_log_level;
    std::atomic<bool>       m_running;
    std::thread             m_thread;
    std::string             m_log_root;
    std::map<std::string, std::shared_ptr<Sink>>  m_log_apps;

    std::mutex                             m_mtx;
    std::condition_variable                m_cond;
    std::list<std::unique_ptr<Record>> m_log_messages;
};

struct tm* LocaltimeEx(const time_t* t, struct tm* output);
void PrintLogFormat(std::string sub_dir_name, LOG_TYPE log_type, const char* format, ...);

void StringPrintf(std::vector<char>* output, const char* format, va_list args);

} //namespace zylib

