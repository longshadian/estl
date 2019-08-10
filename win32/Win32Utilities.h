#pragma once

#include <cstdint>
#include <ctime>
#include <cstring>
#include <string>

class Win32Utilities
{
public:
    static void ThisThreadSleepMilliseconds(std::uint32_t millisec);

    static bool StartService1(const std::string& service_name);
    static bool StopService(const std::string& service_name);
    static bool StartProcess(const std::string& exe_name);
    static void KillProcess(const std::string& exe_name);

    static struct tm* LocaltimeEx(const time_t* t, struct tm* output);
    static std::string LocaltimeYYYMMDD_HHMMSS(std::time_t t);

    static std::string GetExePath();
    static std::string StringFormat(const char *fmt, ...);
};
