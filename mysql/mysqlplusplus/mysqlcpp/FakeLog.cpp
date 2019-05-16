#include "FakeLog.h"

#include <iostream>

namespace mysqlcpp {

std::ostream* g_ostm = nullptr;

void initLog(std::ostream* ostm)
{
    if (!g_ostm)
        g_ostm = ostm;
}

FakeLog::FakeLog(int32_t lv, int line, const char* file, const char* function)
    : m_line(line)
    , m_file(file)
    , m_function(function)
    , m_level(lv)
    , m_stream()
{
}

FakeLog::~FakeLog()
{
    auto s = m_stream.m_ostm.str();
    if (s.empty())
        return;
    if (!g_ostm)
        return;

    if (m_level == LOG_INFO) {
        (*g_ostm) << "[INFO ] [" << m_file << ":" << m_line << "] " << s << "\n";
    } else if (m_level == LOG_ERROR) {
        (*g_ostm) << "[ERROR] [" << m_file << ":" << m_line << "] " << s << "\n";
    }
}

}
