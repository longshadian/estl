#pragma once

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem::v1;

class File
{
public:
    File();
    ~File() = default;

    bool            Open(const char* path);
    unsigned int    Length();
    unsigned int    Read(void* buffer, unsigned int length);
    const std::string& FileName() const;

    static fs::path CreatePath(const char* path);
private:
    fs::path        m_full_path;
    std::string     m_file_name;
};
