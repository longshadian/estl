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
        return static_cast<unsigned int>(fs::file_size(m_path));
    } catch (std::exception& e) {
        (void)e;
        return 0;
    }
}

unsigned int File::Read(void* buffer, unsigned int length)
{
    std::ifstream ifs(m_path);
    ifs.read((char*)buffer, length);
    if (!ifs)
        return 0;
    return length;
    //auto len = ifs.readsome((char*)buffer, length);
    //return (unsigned int)len;
}

fs::path File::CreatePath(const char* path)
{
    return fs::path(path);
}
