#pragma once

#include <string>
#include <vector>
#include <zip.h>

class LibZipTools
{
public:
    LibZipTools();
    ~LibZipTools();

    void Reset();
    bool Uncompress(std::string zip_path, std::string uncompress_path);
    //bool CompressWindowsFile(const std::string& from, const std::string& from_name, const std::string& to, const std::string& to_name);
    bool CompressWindowsFile2(const std::string& from_path, const std::string& from_file, const std::string& to_path, const std::string& to_file);

    // 压缩单个文件
    bool CompressFile(const std::string& from_path, const std::string& from_file, const std::string& to_path, const std::string& to_file);
    bool CompressFile(const std::string& from_path, const std::vector<std::string>& file_list, const std::string& to_path, const std::string& to_file);

    // 压缩目录
    bool CompressDir(const std::string& from_path, const std::string& to_path, const std::string& to_file);

private:
    zip_t* m_zipfile;
};
