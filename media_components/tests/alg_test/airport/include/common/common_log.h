//
// Created by pengguanhai on 2020/10/15.
//

#ifndef AIRPORT_TRACKING_COMMON_LOG_H
#define AIRPORT_TRACKING_COMMON_LOG_H

enum LOG_LEVEL
{
    Debug	= 0,
    Info	= 1,
    Warning = 2,
    Error	= 3,
    NUM_SEVERITY = 4,
};

using LogCallback = void (*)(LOG_LEVEL severity, const char* msg);
void SetLogCallback(LogCallback cb);

void SetLogLevel(LOG_LEVEL severity);

void Print(LOG_LEVEL severity, const char* fmt, ...);

#define CPP_LOG(level, format, ...) Print(level, "%d:%s:%s " format, __LINE__, __FILE__, __FUNCTION__, ##__VA_ARGS__)

#endif  // AIRPORT_TRACKING_COMMON_LOG_H
