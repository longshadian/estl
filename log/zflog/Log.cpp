#include "Log.h"

#include <array>
#include <cstdio>
#include <cstring>

namespace yzhl
{

enum {DEFAUL_BUFFER_LENGTH = 1024 * 1};         // 1K
enum {MAX_BUFFER_LENGTH = 1024 * 64  };     	// 64K


/**
 * class record
 **************************************************************************/
Record::Record()
    : m_type(0)
    , m_dir_name()
    , m_buffer(DEFAUL_BUFFER_LENGTH, 0)
{
}


/**
 * class Sink
 **************************************************************************/
Sink::Sink(std::string log_root, std::string dir_name)
    : m_file(nullptr)
    , m_day(0)
    , m_dir_name()
{
    time_t cur_time = time(NULL);
    struct tm cur_tm;
    LocaltimeEx(&cur_time, &cur_tm);
    m_day = cur_tm.tm_mday;
    m_dir_name = log_root + "/" + dir_name;
    if (m_dir_name.back() != '/')
        m_dir_name.push_back('/');
}

Sink::~Sink()
{
    if (m_file)
        ::fclose(m_file);
}

void Sink::flush(const char* log_type, const char* log_content, size_t length)
{
    time_t t = time(NULL);
    struct tm cur_tm;
    LocaltimeEx(&t, &cur_tm);
    if (cur_tm.tm_mday != m_day) {
        m_day = cur_tm.tm_mday;
        if (m_file)
            ::fclose(m_file);
        m_file = NULL;
    }

    std::array<char, 128> log_head{0};
    snprintf(log_head.data(), log_head.size(), "[%s] %02d:%02d:%02d\t", log_type, cur_tm.tm_hour, cur_tm.tm_min, cur_tm.tm_sec);
    if (!m_file) {
        std::array<char, 256> file_name{0};
        snprintf(file_name.data(), file_name.size(), "%s/%04d%02d%02d.log", m_dir_name.c_str(), cur_tm.tm_year + 1900, cur_tm.tm_mon + 1, cur_tm.tm_mday);
        m_file = ::fopen(file_name.data(), "a+");
    }
    if (m_file) {
        ::fwrite(log_head.data(), 1, strlen(log_head.data()), m_file);
        ::fwrite(log_content, 1, length, m_file);
        /*
        char c = '\n';
        ::fwrite(&c, 1, 1, m_file);
        */
        ::fflush(m_file);
    }
}

/**
 * class Logger
 **************************************************************************/
Logger::Logger() 
	: m_log_level(LOG_DEBUG | LOG_INFO | LOG_WARNING | LOG_ERROR)
    , m_running(false)
    , m_thread()
    , m_log_root()
    , m_log_apps()
{
}

Logger::~Logger()
{
    m_running = false;
    pushMsg(nullptr);
    if (m_thread.joinable())
        m_thread.join();
}

Logger& Logger::Get()
{
    static Logger _instance;
    return _instance;
}

void Logger::pushMsg(std::unique_ptr<Record> msg)
{
    std::lock_guard<std::mutex> lk(m_mtx);
    m_log_messages.push_back(std::move(msg));
    m_cond.notify_one();
}

void Logger::setLogLevel(int level)
{
    m_log_level = level;
}

int Logger::getLogLevel() const
{
    return m_log_level;
}

void Logger::StartThread()
{
    std::unique_ptr<Record> msg = nullptr;
    while(m_running) {
        {
            std::unique_lock<std::mutex> lk(m_mtx);
            m_cond.wait(lk, [this] { return !m_log_messages.empty(); });
            msg = std::move(m_log_messages.front());
            m_log_messages.pop_front();
        }
        if (!msg) {
            continue;
        }

        std::shared_ptr<Sink> app = nullptr;
        auto it = m_log_apps.find(msg->m_dir_name);
        if (it == m_log_apps.end()) {
            if (!msg->m_dir_name.empty()) {
                app = std::make_shared<Sink>(m_log_root, msg->m_dir_name);
                m_log_apps[msg->m_dir_name] = app;
            }
        } else {
            app = it->second;
        }

        if (app) {
            const char* log_type = nullptr;
            switch(msg->m_type) {
            case LOG_DEBUG:
                log_type = "DEBUG";
                break;
            case LOG_WARNING:
                log_type = "TJ";
                break;
            case LOG_INFO:
                log_type = "TIP";
                break;
            case LOG_ERROR:
                log_type = "ERROR";
            default:
                log_type = "UNKONWN";
                break;
            }
            if (msg->m_buffer.empty()) {
                msg->m_buffer.push_back('\n');
            } else {
                *msg->m_buffer.rbegin() = '\n';
            }
            app->flush(log_type, msg->m_buffer.data(), msg->m_buffer.size());
        }
    }
}

void Logger::init(std::string path)
{
    Get().start(std::move(path));
}

void Logger::start(std::string path_end_with_sprit)
{
    m_log_root = std::move(path_end_with_sprit);
    if (m_log_root.empty())
        return;
    if (m_log_root.back() != '/')
        m_log_root.push_back('/');

    m_running = true;
    std::thread temp(std::bind(&Logger::StartThread, this));
    m_thread = std::move(temp);
}



struct tm* LocaltimeEx(const time_t* t, struct tm* output)
{
#ifdef WIN32
    localtime_s(output, t);
#else
    localtime_r(t, output);
#endif
    return output;
}

void PrintLogFormat(std::string sub_dir_name, LOG_TYPE log_type, const char* format, ...)
{
	if ((Logger::Get().getLogLevel() & log_type) == 0)
		return;

    std::unique_ptr<Record> log_msg(new Record());
    log_msg->m_type = log_type;
    log_msg->m_dir_name = std::move(sub_dir_name);

    va_list va;
    va_start(va, format);
    StringPrintf(&log_msg->m_buffer, format, va);
    va_end(va);
    Logger::Get().pushMsg(std::move(log_msg));
}

void StringPrintf(std::vector<char>* output, const char* format, va_list args)
{
    size_t remaining = DEFAUL_BUFFER_LENGTH;
    output->resize(DEFAUL_BUFFER_LENGTH);

    va_list args_copy;
    va_copy(args_copy, args);
    int bytes_used = vsnprintf(output->data(), remaining, format, args_copy);
    va_end(args_copy);
    if (bytes_used < 0) {
        output->clear();
        return;
    }

    if (static_cast<std::size_t>(bytes_used) < remaining) {
        output->resize(bytes_used);
        return;
    }

    if (bytes_used + 1 > MAX_LOG_MSG_LENGTH) {
        output->resize(MAX_LOG_MSG_LENGTH);
        remaining = MAX_LOG_MSG_LENGTH;
    } else {
        output->resize(bytes_used + 1);
        remaining = bytes_used + 1;
    }

    va_list args_copy;
    va_copy(args_copy, args);
    bytes_used = vsnprintf(output->data(), remaining, format, args_copy);
    va_end(args_copy);
    if (bytes_used < 0 ) {
        output->clear();
    }
}

} // namespace yzhl

