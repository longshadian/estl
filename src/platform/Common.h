#pragma once

#include <cstdio>

#if 0
#define APrintf(severity, fmt, ...) \
    do { \
        printf("[%s] [%s] [line:%04d] " fmt "\n", zylib::Localtime_HHMMSS_F().c_str(), severity, __LINE__, ##__VA_ARGS__); \
    } while (0)
#else
#define APrintf(severity, fmt, ...) \
    do { \
        printf("[%s] [line:%d] [%s] " fmt "\n", severity, __LINE__, __FILE__, ##__VA_ARGS__); \
    } while (0)
#endif

#define DPrintf(fmt, ...) APrintf("DEBUG  ", fmt, ##__VA_ARGS__)
#define WPrintf(fmt, ...) APrintf("WARNING", fmt, ##__VA_ARGS__)
