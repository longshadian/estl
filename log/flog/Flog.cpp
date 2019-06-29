#include "Flog.h"

/*
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
*/

#include <cstdio>
#include <cstring>

namespace flog
{

LogBuffer::LogBuffer()
{
}

LogBuffer::~LogBuffer()
{
}

LogBuffer::LogBuffer(LogBuffer&& rhs)
{
}

LogBuffer& LogBuffer::operator=(LogBuffer&& rhs)
{
}

void LogBuffer::Resize(std::size_t length)
{

}

std::uint8_t* LogBuffer::WritePtr()
{

}

const std::uint8_t* LogBuffer::ReadPtr() const
{

}

void LogBuffer::Consume(std::size_t length)
{
}


Logger::LogApp::LogApp(const char* log_root, const char* app_name)
{
    m_file = NULL;
    memset(m_app_name, 0, sizeof(m_app_name));
    m_day = 0;

    time_t curTime = time(NULL);
    struct tm cur_tm;
    localtime_r(&curTime, &cur_tm);
    m_day = cur_tm.tm_mday;

    std::string app_log_dir = log_root;
    app_log_dir += "/";
    app_log_dir += app_name;
    strncpy(m_app_name, app_log_dir.c_str(), sizeof(m_app_name));
    mkdir(m_app_name, S_IWUSR | S_IXUSR | S_IRUSR);
}

Logger::LogApp::~LogApp()
{
    if (m_file)
        fclose(m_file);
}

void Logger::LogApp::log(const char* log_type, const char* log_content)
{
    time_t t = time(NULL);
    struct tm cur_tm;
    localtime_r(&t, &cur_tm);
    if (cur_tm.tm_mday != m_day) {
        m_day = cur_tm.tm_mday;
        if (m_file)
            fclose(m_file);
        m_file = NULL;
    }

    char log_head[64];
    snprintf(log_head, sizeof(log_head), "[%s] %02d:%02d:%02d\t",
        log_type, cur_tm.tm_hour, cur_tm.tm_min, cur_tm.tm_sec);
    if (!m_file) {
        char file_name[256];
        snprintf(file_name, sizeof(file_name), "%s/%04d%02d%02d.log",
            m_app_name, cur_tm.tm_year + 1900, cur_tm.tm_mon + 1, cur_tm.tm_mday);
        m_file = fopen(file_name, "a+");
    }
    if (m_file) {
        fwrite(log_head, 1, strlen(log_head), m_file);
        fwrite(log_content, 1, strlen(log_content), m_file);
        char c = '\n';
        fwrite(&c, 1, 1, m_file);
        fflush(m_file);
    }
}

//////////////////////////////////////////////////////////////////////////
Logger::Logger()
    : m_log_level(LOG_DEBUG | LOG_TIP | LOG_TJ | LOG_ERROR | LOG_TRACE)
    , m_running(false)
    , m_thread()
    , m_log_root()
    , m_log_queue()
    , m_log_apps()
{

    //假设路径 /home/xxx/server_install_path/bin/server
    //                                     /log

    //m_log_root = getRootDir();

    std::string full_app_name = getFullAppName();

    if (full_app_name.length() > 0) {
        int count = 0;
        auto it = full_app_name.rbegin();
        for (; it != full_app_name.rend(); ++it) {
            if (*it == '/')
                count++;

            //去除 bin/server
            if (count == 2) {
                break;
            }
        }
        std::string temp(it, full_app_name.rend());
        m_log_root.assign(temp.rbegin(), temp.rend());
        m_log_root += "log";
    }

    if (!m_log_root.empty()) {
        mkdir(m_log_root.c_str(), S_IWUSR | S_IXUSR | S_IRUSR);
    }
}

Logger::~Logger()
{
    m_running = false;
    m_log_queue.push(log_t());
    if (m_thread.joinable())
        m_thread.join();
}

Logger& Logger::Get()
{
    static Logger _instance;
    return &_instance;
}

void Logger::Post(LogRecordUPtr record)
{
    std::lock_guard<std::mutex> lk(m_mtx);
    m_list.push_back(std::move(record));
    m_cond.notify_all();
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
    LogRecordUPtr record;
    while (m_running) {
        {
            std::unique_lock<std::mutex> lk(m_mtx);
            m_cond.wait_for(lk, std::chrono::seconds(2), [this] { return !m_list.empty()});
            if (!m_list.empty()) {
                record = std::move(m_list.front());
                m_list.pop_front();
            }
        }
        if (!record) {
            continue;
        }

        WriteRecord(std::move(record));
        record.reset();

    }
}

std::string Logger::getFullAppName()
{
    char full_app_name[256] = { 0 };
    getexepath(full_app_name, sizeof(full_app_name));
    if (strlen(full_app_name) > 0)
        return full_app_name;
    return std::string();
}

std::string Logger::getRootDir()
{
    std::string full_path;
    char full_app_name[256] = { 0 };
    getexepath(full_app_name, sizeof(full_app_name));

    if (strlen(full_app_name) > 0) {
        char* name = strrchr(full_app_name, '/');
        if (name) {
            *name = 0;
            name = strrchr(full_app_name, '/');
            if (name) {
                name++;
                *name = 0;
                full_path = full_app_name;
            }
        }
    }
    return full_path + "log";
}

int Logger::getexepath(char* path, int max)
{
    char buf[128];
    sprintf(buf, "/proc/%d/exe", getpid());
    int n = static_cast<int>(readlink(buf, path, max));
    return n;
}

void Logger::WriteRecord(LogRecordUPtr record)
{
    std::shared_ptr<LogRecord> app;
    auto it = m_log_apps.find(msg.m_app_name);
    if (it == m_log_apps.end()) {
        if (strlen(msg.m_app_name) > 0) {
            app = std::make_shared<LogRecord>(m_log_root.c_str(), msg.m_app_name);
            m_log_apps[msg.m_app_name] = app;
        }
    }
    else {
        app = it->second;
    }

    if (app) {
        char log_type[32];
        switch (msg.m_type) {
        case LOG_TRACE:
            strcpy(log_type, "TRACE");
            break;
        case LOG_DEBUG:
            strcpy(log_type, "DEBUG");
            break;
        case LOG_TJ:
            strcpy(log_type, "TJ   ");
            break;
        case LOG_TIP:
            strcpy(log_type, "TIP  ");
            break;
        default:
            strcpy(log_type, "ERROR");
            break;
        }
        app->log(log_type, msg.m_app_log);
    }

    time_t t = time(NULL);
    struct tm cur_tm;
    localtime_r(&t, &cur_tm);
    if (cur_tm.tm_mday != m_day) {
        m_day = cur_tm.tm_mday;
        if (m_file)
            fclose(m_file);
        m_file = NULL;
    }

    char log_head[64];
    snprintf(log_head, sizeof(log_head), "[%s] %02d:%02d:%02d\t",
        log_type, cur_tm.tm_hour, cur_tm.tm_min, cur_tm.tm_sec);
    if (!m_file) {
        char file_name[256];
        snprintf(file_name, sizeof(file_name), "%s/%04d%02d%02d.log",
            m_app_name, cur_tm.tm_year + 1900, cur_tm.tm_mon + 1, cur_tm.tm_mday);
        m_file = fopen(file_name, "a+");
    }
    if (m_file) {
        fwrite(log_head, 1, strlen(log_head), m_file);
        fwrite(log_content, 1, strlen(log_content), m_file);
        char c = '\n';
        fwrite(&c, 1, 1, m_file);
        fflush(m_file);
    }


}

void Logger::Init()
{
    m_running = true;
    std::thread temp(std::bind(&Logger::StartThread, this));
    m_thread = std::move(temp);
}

void PRINT_LOG(const char* app_name, int log_type, const char* log_content)
{
    if ((Logger::Get().getLogLevel() & log_type) == 0)
        return;

    Logger::log_t one_log;
    memset(&one_log, 0, sizeof(one_log));
    strncpy(one_log.m_app_name, app_name, sizeof(one_log.m_app_name) - 1);
    one_log.m_type = log_type;
    strncpy(one_log.m_app_log, log_content, strlen(log_content));
    Logger::Get().log(one_log);
}

void PRINT_LOG_EX(const char* app_name, int log_type, const char* format, ...)
{
    if ((Logger::Get().getLogLevel() & log_type) == 0)
        return;

    Logger::log_t one_log;
    memset(&one_log, 0, sizeof(one_log));
    va_list va;
    va_start(va, format);
    vsnprintf(one_log.m_app_log, sizeof(one_log.m_app_log) - 1, format, va);
    va_end(va);
    strncpy(one_log.m_app_name, app_name, sizeof(one_log.m_app_name) - 1);
    one_log.m_type = log_type;
    Logger::Get().log(one_log);
}

void PrintFormat(std::string sub_dir_name, LOG_TYPE log_type, const char* format, ...)
{
    std::unique_ptr<LogRecord> record = CreateRecord();
    record->m_log_type = log_type;
    record->m_dir_name = std::move(sub_dir_name);

    va_list va;
    va_start(va, format);
    StringPrintf(record, format, va);
    va_end(va);
    ServerLoger::getInstance().pushMsg(std::move(record));
}

std::unique_ptr<LogRecord> CreateRecord()
{
    return std::make_unique<LogRecord>();
}

void StringPrintf(LogRecord* output, const char* format, va_list args)
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
    }
    else if ((std::vector<char>::size_type)bytes_used < remaining) {
        output->resize(bytes_used);
    }
    else {
        if (bytes_used + 1 > MAX_LOG_MSG_LENGTH) {
            output->resize(MAX_LOG_MSG_LENGTH);
            remaining = MAX_LOG_MSG_LENGTH;
        }
        else {
            output->resize(bytes_used + 1);
            remaining = bytes_used + 1;
        }

        va_list args_copy;
        va_copy(args_copy, args);
        bytes_used = vsnprintf(output->data(), remaining, format, args_copy);
        va_end(args_copy);
        if (bytes_used < 0) {
            output->clear();
        }
    }
}

} // namespace flog
