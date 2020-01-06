#define APrintf(severity, fmt, ...) \
    do { \
        printf("[%s] [%s] [line:%04d] " fmt "\n", zylib::Localtime_HHMMSS_F().c_str(), severity, __LINE__, ##__VA_ARGS__); \
    } while (0)

#define DPrintf(fmt, ...) APrintf("DEBUG  ", fmt, ##__VA_ARGS__)
#define WPrintf(fmt, ...) APrintf("WARNING", fmt, ##__VA_ARGS__)