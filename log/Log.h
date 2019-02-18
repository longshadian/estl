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

namespace zylib {

enum LOG_TYPE
{
	LOG_DEBUG    = 1,
	LOG_TIP      = 2,
	LOG_TJ       = 4,
	LOG_ERROR    = 8
};

class ServerLoger
{
    class Dir
    {
    public:
        Dir(std::string log_root, std::string dir_name);
        ~Dir();
        void flush(const char* log_type, const char* log_content, size_t length);
    protected:
        FILE*       m_file;
        int	        m_day;
        std::string	m_dir_name;
    };

    ServerLoger();
public:
    enum {MIN_LOG_MSG_LENGTH = 128};
    enum {MAX_LOG_MSG_LENGTH = 1024};

    struct LogMessage
    {
        LogMessage();
        int m_type;
        std::string m_dir_name;
        std::vector<char> m_buffer;
    };
public:
    ~ServerLoger();
    static ServerLoger& getInstance();

    static void init(std::string path);
    static void printLogFormat(std::string sub_dir_name, LOG_TYPE log_type, const char* format, ...);

    void setLogLevel(int level);
    int getLogLevel() const;
private:
    void start(std::string path_end_with_sprit);
    void run();
    void pushMsg(std::unique_ptr<LogMessage> msg);

    static struct tm* localtimeEx(const time_t* t, struct tm* output);
    static void stringPrintf(std::vector<char>* output, const char* format, va_list args);
private:
    int                     m_log_level;
    std::atomic<bool>       m_running;
    std::thread             m_thread;
    std::string             m_log_root;
    std::map<std::string, std::shared_ptr<Dir>>  m_log_apps;

    std::mutex                             m_mtx;
    std::condition_variable                m_cond;
    std::list<std::unique_ptr<LogMessage>> m_log_messages;
};

}

#endif
