#pragma once

#include <sstream>
#include <memory>
#include <mutex>

namespace network {

class LogStream
{
public:
    LogStream();
    virtual ~LogStream();
public:
    LogStream& operator<<(signed char val);
    LogStream& operator<<(unsigned char val);
    LogStream& operator<<(char val);
    LogStream& operator<<(short val);
    LogStream& operator<<(unsigned short val);
    LogStream& operator<<(int val);
    LogStream& operator<<(unsigned int val);
    LogStream& operator<<(long val);
    LogStream& operator<<(unsigned long val);
    LogStream& operator<<(long long val);
    LogStream& operator<<(unsigned long long val);
    LogStream& operator<<(float val);
    LogStream& operator<<(double val);
    LogStream& operator<<(long double val);
    LogStream& operator<<(const std::string& val);
    LogStream& operator<<(const char* val);

    virtual void flush();

    std::mutex         m_mtx;
    std::ostringstream m_ostm;
};

enum LOG_LEVEL
{
	DEBUG	= 0,
	INFO	= 1,
	WARNING = 2,
	ERROR	= 3,
	NUM_SEVERITY = 4,
};

extern const char* LOG_SEVERITY_NAMES[NUM_SEVERITY];

void initLog(std::unique_ptr<LogStream> ostm, LOG_LEVEL lv = DEBUG);

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
    FakeLog(int lv, int line, const char* file, const char* function);
    FakeLog(int lv, const char* file, int line);
    ~FakeLog();

    FakeLogStream& stream() { return m_stream; }
private:
    int             m_line;
    const char*     m_file;
    const char*     m_function;
    int				m_level;
    FakeLogStream   m_stream;
};

#define FAKE_LOG(type)	FakeLog(type,		__FILE__, __LINE__).stream()

}
