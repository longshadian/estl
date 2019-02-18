#include "Log.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <sys/time.h>

namespace zylib {

const char* getLogSeverityName(LOG_SEVERITY s)
{
    static std::array<const char*, FATAL + 1> names = 
    {
        "DEBUG  ",
        "TIP    ", 
        "WARNING",
        "ERROR  ",
        "FATAL  ",
    };
    return names[s];
}

std::string LogStream::getString()
{
    return m_ostm.str();
}


Logger::Logger(LOG_SEVERITY s)
    : m_severity(s)
    , m_stream()
{
}

Logger::~Logger()
{
    m_stream << '\n';
    ServerLogger::getInstance().printLog(m_severity, m_stream.getString());
}

LogStream& Logger::stream()
{
    return m_stream;
}

ServerLogger::LogFile::LogFile(std::string path, std::string file_name)
    : m_file(nullptr)
    , m_day(0)
    , m_full_file_name()
{
    time_t cur_time = time(nullptr);
    struct tm cur_tm;
    localtimeEx(&cur_time, &cur_tm);
    m_day = cur_tm.tm_mday;
    m_full_file_name = path + "/" + file_name;
}

ServerLogger::LogFile::~LogFile()
{
    if (m_file)
        ::fclose(m_file);
}

void ServerLogger::LogFile::flush(const char* severity_name, const char* log_content, size_t length)
{
    struct timeval tv{};
    gettimeofday(&tv, nullptr);
    time_t t = time_t(tv.tv_sec);
    struct tm cur_tm{};
    localtimeEx(&t, &cur_tm);
    if (cur_tm.tm_mday != m_day) {
        m_day = cur_tm.tm_mday;
        if (m_file)
            ::fclose(m_file);
        m_file = NULL;
    }

    std::array<char, 128> log_head{0};
    auto head_len = snprintf(log_head.data(), log_head.size(), "[%s] %02d:%02d:%02d.%06d ", severity_name, cur_tm.tm_hour, cur_tm.tm_min, cur_tm.tm_sec, (int)tv.tv_usec);
    if (!m_file) {
        std::array<char, 256> file_name{0};
        snprintf(file_name.data(), file_name.size(), "%s/%04d%02d%02d.log", m_full_file_name.c_str(), cur_tm.tm_year + 1900, cur_tm.tm_mon + 1, cur_tm.tm_mday);
        m_file = ::fopen(file_name.data(), "a+");
    }
    if (m_file) {
        ::fwrite(log_head.data(), 1, head_len, m_file);
        if (length > 0)
            ::fwrite(log_content, 1, length, m_file);
        ::fflush(m_file);
    }
}

//////////////////////////////////////////////////////////////////////////
ServerLogger::ServerLogger() 
    : m_path()
    , m_file_name()
    , m_log_severity(DEBUG)
    , m_log_file(nullptr)
    , m_running(false)
    , m_thread()
    , m_mtx()
    , m_cond()
    , m_log_messages()
{
}

ServerLogger::~ServerLogger()
{
    m_running = false;
    pushMsg(nullptr);
    if (m_thread.joinable())
        m_thread.join();
}

ServerLogger& ServerLogger::getInstance()
{
    static ServerLogger _instance;
    return _instance;
}

void ServerLogger::pushMsg(std::unique_ptr<LogMessage> msg)
{
    std::lock_guard<std::mutex> lk(m_mtx);
    m_log_messages.push_back(std::move(msg));
    m_cond.notify_one();
}

LOG_SEVERITY ServerLogger::getLogSeverity() const
{
    return m_log_severity;
}

void ServerLogger::run()
{
    std::unique_ptr<LogMessage> msg = nullptr;
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

        if (m_log_file) {
            if (!msg->m_buffer.empty())
                m_log_file->flush(getLogSeverityName(msg->m_severity), msg->m_buffer.data(), msg->m_buffer.size());
            if (!msg->m_content.empty())
                m_log_file->flush(getLogSeverityName(msg->m_severity), msg->m_content.data(), msg->m_content.size());
        }
    }
}

void ServerLogger::init(std::string path, std::string file_name, LOG_SEVERITY s)
{
    getInstance().start(std::move(path), std::move(file_name), s);
}

void ServerLogger::start(std::string path_end_with_sprit, std::string file_name, LOG_SEVERITY s)
{
    m_path = std::move(path_end_with_sprit);
    if (m_path.empty())
        return;
    m_file_name = std::move(file_name);
    m_log_file = std::make_shared<LogFile>(m_path, m_file_name);

    m_running = true;
    std::thread temp(std::bind(&ServerLogger::run, this));
    m_thread = std::move(temp);
}

void ServerLogger::printLogFormat(LOG_SEVERITY log_severity, const char* format, ...)
{
	if (ServerLogger::getInstance().getLogSeverity() > log_severity)
		return;

    std::unique_ptr<ServerLogger::LogMessage> log_msg{ new ServerLogger::LogMessage() };
    log_msg->m_severity = log_severity;

    va_list va;
    va_start(va, format);
    stringPrintf(&log_msg->m_buffer, format, va);
    va_end(va);
    log_msg->m_buffer.push_back('\n');
    ServerLogger::getInstance().pushMsg(std::move(log_msg));
}

void ServerLogger::printLog(LOG_SEVERITY log_severity, std::string s)
{
    if (ServerLogger::getInstance().getLogSeverity() > log_severity)
        return;

    std::unique_ptr<ServerLogger::LogMessage> log_msg{ new ServerLogger::LogMessage() };
    log_msg->m_severity = log_severity;
    log_msg->m_content = std::move(s);
    ServerLogger::getInstance().pushMsg(std::move(log_msg));
}

void ServerLogger::stringPrintf(std::vector<char>* output, const char* format, va_list args)
{
    size_t remaining = MIN_LOG_MSG_LENGTH;
    output->resize(MIN_LOG_MSG_LENGTH);

    va_list args_copy;
    va_copy(args_copy, args);
    int bytes_used = vsnprintf(output->data(), remaining, format, args_copy);
    va_end(args_copy);
    if (bytes_used < 0) {
        output->clear();
        return;
    } else if ((std::vector<char>::size_type)bytes_used < remaining) {
        output->resize(bytes_used);
    } else {
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
}

struct tm* ServerLogger::localtimeEx(const time_t* t, struct tm* output)
{
#ifdef WIN32
    localtime_s(output, t);
#else
    localtime_r(t, output);
#endif
    return output;
}

}

zylib::LogStream& operator<<(zylib::LogStream& os, bool v) { os.m_ostm << v; return os; }
zylib::LogStream& operator<<(zylib::LogStream& os, char v) { os.m_ostm << v; return os; }
zylib::LogStream& operator<<(zylib::LogStream& os, signed char v) { os.m_ostm << v; return os; }
zylib::LogStream& operator<<(zylib::LogStream& os, unsigned char v) { os.m_ostm << v; return os; }
zylib::LogStream& operator<<(zylib::LogStream& os, short v) { os.m_ostm << v; return os; }
zylib::LogStream& operator<<(zylib::LogStream& os, unsigned short v) { os.m_ostm << v; return os; }
zylib::LogStream& operator<<(zylib::LogStream& os, int v) { os.m_ostm << v; return os; }
zylib::LogStream& operator<<(zylib::LogStream& os, unsigned int v) { os.m_ostm << v; return os; }
zylib::LogStream& operator<<(zylib::LogStream& os, long v) { os.m_ostm << v; return os; }
zylib::LogStream& operator<<(zylib::LogStream& os, unsigned long v) { os.m_ostm << v; return os; }
zylib::LogStream& operator<<(zylib::LogStream& os, long long v) { os.m_ostm << v; return os; }
zylib::LogStream& operator<<(zylib::LogStream& os, unsigned long long v) { os.m_ostm << v; return os; }
zylib::LogStream& operator<<(zylib::LogStream& os, float v) { os.m_ostm << v; return os; }
zylib::LogStream& operator<<(zylib::LogStream& os, double v) { os.m_ostm << v; return os; }
zylib::LogStream& operator<<(zylib::LogStream& os, long double v) { os.m_ostm << v; return os; }
zylib::LogStream& operator<<(zylib::LogStream& os, std::string v) { os.m_ostm << v; return os; }
zylib::LogStream& operator<<(zylib::LogStream& os, const char* v) { os.m_ostm << v; return os; }


