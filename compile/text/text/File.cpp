#include "File.h"

#include <experimental/filesystem>
#include <fstream>

File::File()
    : m_full_path()
    , m_file_name()
{
}

bool File::Open(const char* path)
{
    try {
        m_full_path = fs::path(path);
        if (m_full_path.has_filename()) {
            m_file_name = m_full_path.filename().generic_string();
        }
        return true;
    } catch (std::exception& e) {
        (void)e;
        return false;
    }
}

unsigned int File::Length()
{
    try {
        return static_cast<unsigned int>(fs::file_size(m_full_path));
    } catch (std::exception& e) {
        (void)e;
        return 0;
    }
}

unsigned int File::Read(void* buffer, unsigned int length)
{
    std::ifstream ifs(m_full_path);
    ifs.read((char*)buffer, length);
    if (!ifs)
        return 0;
    return length;
    //auto len = ifs.readsome((char*)buffer, length);
    //return (unsigned int)len;
}

const std::string& File::FileName() const
{
    return m_file_name;
}

fs::path File::CreatePath(const char* path)
{
    return fs::path(path);
}

