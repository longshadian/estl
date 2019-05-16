#pragma once

#include <sstream>

namespace mysqlcpp {

enum LOG_LEVEL
{
	Debug	= 0,
	Info	= 1,
	Warning = 2,
	Error	= 3,
	NUM_SEVERITY = 4,
};

using LogCallback = void (*)(LOG_LEVEL severity, const char* msg);
void MYSQLCPP_EXPORT SetLogCallback(LogCallback cb);

void MYSQLCPP_EXPORT SetLogLevel(LOG_LEVEL severity);

void Print(LOG_LEVEL severity, const char* fmt, ...);
//void VerbosePrint(int line, const char* file, LOG_LEVEL severity, const char* fmt, ...);

}

//#define MYSQLCPP_LOG(log_severity, fmt, ...) mysqlcpp::Print(severity, fmt, ...)
#define MYSQLCPP_LOG(level, format, ...) Print(level, "%d:%s:%s " format, __LINE__, __FILE__, __FUNCTION__, ##__VA_ARGS__)

