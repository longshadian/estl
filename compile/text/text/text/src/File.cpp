#include "File.h"

#include <experimental/filesystem>
#include <fstream>

File::File()
    : m_path()
{

}

bool File::Open(const char* path)
{
    try {
        m_path = fs::path(path);
        return true;
    } catch (std::exception& e) {
        (void)e;
        return false;
    }
}

unsigned int File::Length()
{
    try {
        return fs::file_size(m_path);
    } catch (std::exception& e) {
        (void)e;
        return 0;
    }
}

unsigned int File::Read(void* buffer, unsigned int length)
{
    m_path.c_str();
    std::ifstream ifs(m_path);
    auto len = ifs.readsome((char*)buffer, length);
    return (unsigned int)len;
}

fs::path File::CreatePath(const char* path)
{
    return fs::path(path);
}
