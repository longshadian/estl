#include "mysqlcpp/MysqlcppLog.h"

#include <cstdarg>
#include <iostream>

namespace mysqlcpp {

static LogCallback g_log_function = nullptr;
static LOG_LEVEL g_log_severity = LOG_LEVEL::Debug;
    
void SetLogCallback(LogCallback cb)
{
    g_log_function = cb;
}

static void PrintLog(LOG_LEVEL severity, const char *msg)
{
    if (g_log_function) {
        g_log_function(severity, msg);
    } else {
        const char *severity_str;
        switch (severity) {
        case LOG_LEVEL::Debug:
            severity_str = "MYSQLCPP_DEBUG";
            break;
        case LOG_LEVEL::Info:
            severity_str = "MYSQLCPP_INFO";
            break;
        case LOG_LEVEL::Warning:
            severity_str = "MYSQLCPP_WARNING";
            break;
        case LOG_LEVEL::Error:
            severity_str = "MYSQLCPP_ERROR";
            break;
        default:
            severity_str = "???";
            break;
        }
        (void)fprintf(stderr, "[%s] %s\n", severity_str, msg);
    }
}

static int VSNPrintf(char* dest, int size, const char* fmt, va_list argptr)
{
    int ret = vsnprintf(dest, size - 1, fmt, argptr);
    dest[size - 1] = '\0';
    if (ret < 0 || ret >= size) {
        return -1;
    }
    return ret;
}

void SetLogLevel(LOG_LEVEL severity)
{
    g_log_severity = severity;
}

void Print(LOG_LEVEL severity, const char* fmt, ...)
{
    if (severity < g_log_severity)
        return;

    va_list argptr;
    char msg[1024];

    va_start(argptr, fmt);
    VSNPrintf(msg, sizeof(msg), fmt, argptr);
    va_end(argptr);
    msg[sizeof(msg) - 1] = '\0';
    PrintLog(severity, msg);
}

/*
void VerbosePrint(int line, const char* file, LOG_LEVEL severity, const char* fmt, ...)
{
    if (severity < g_log_severity)
        return;

    va_list argptr;
    char msg[1024];

    va_start(argptr, fmt);
    VSNPrintf(msg, sizeof(msg), fmt, argptr);
    va_end(argptr);
    msg[sizeof(msg) - 1] = '\0';
    PrintLog(severity, msg);
}
*/

}
