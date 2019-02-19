#ifndef _MYSQLCPP_FAKELOG_H
#define _MYSQLCPP_FAKELOG_H

#include <sstream>

namespace mysqlcpp {

void initLog(std::ostream* ostm);

struct FakeLogStream
{
    std::ostringstream m_ostm;
};

template <typename T>
inline FakeLogStream& operator<<(FakeLogStream& os, const T& t)
{
    (void)t;
    os.m_ostm << t;
    return os;
};

struct FakeLog
{
    enum LOG_LEVEL
    {
        LOG_ERROR = 0,
        LOG_INFO = 1,
    };

    FakeLog(int32_t lv, int line, const char* file, const char* function);
    ~FakeLog();

    FakeLogStream& stream() { return m_stream; }
private:
    int             m_line;
    const char*     m_file;
    const char*     m_function;
    int32_t         m_level;
    FakeLogStream   m_stream;
};


#define FAKE_LOG_INFO() FakeLog(FakeLog::LOG_INFO, __LINE__, __FILE__, __FUNCTION__).stream()
#define FAKE_LOG_ERROR() FakeLog(FakeLog::LOG_ERROR, __LINE__, __FILE__, __FUNCTION__).stream()
#define FAKE_LOG_WARRING() FakeLog(FakeLog::LOG_ERROR, __LINE__, __FILE__, __FUNCTION__).stream()

}


#endif
