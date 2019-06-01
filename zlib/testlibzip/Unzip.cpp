#include "Unzip.h"

#include <array>
#include <cstdio>
#include <boost/filesystem.hpp>
#include <zip.h>

#define PrintLog printf

Unzip::Unzip()
    : m_zipfile(nullptr)
{
}

Unzip::~Unzip()
{
    if (m_zipfile) {
        ::zip_close(m_zipfile);
        m_zipfile = nullptr;
    }
}

bool Unzip::Uncompress(std::string zipfile_path, std::string uncompress_path)
{
    int error_code = 0;

    std::array<char, 1024> buf{};
    struct zip_stat stat;

    m_zipfile = ::zip_open(zipfile_path.c_str(), ZIP_CHECKCONS, &error_code);
    if (!m_zipfile) {
        PrintLog("ERROR: zip open failed:%d\n", error_code);
        return false;
    }

    zip_int64_t num_enties = ::zip_get_num_entries(m_zipfile, 0);
    for (zip_int64_t i = 0; i < num_enties; i++) {
        std::memset(&stat, 0, sizeof(stat));
        if (::zip_stat_index(m_zipfile, i, 0, &stat) == 0) {
            //LogPrintf("file name is:%s \t\t index: %d size: %d com_size: %d\n", stat.name, stat.index, stat.size, stat.comp_size);
        }

        try {
            std::string entry_name = stat.name;
            boost::filesystem::path entry_path(uncompress_path);
            entry_path = entry_path / entry_name;
            std::string entry_path_str = entry_path.generic_string();
            // TODO 更好的判断目录??
            if (*entry_name.rbegin() == '/') {
                if (!boost::filesystem::exists(entry_path)) {
                    if (!boost::filesystem::create_directory(entry_path)) {
                        PrintLog("ERROR: create dir: %s failed.\n", entry_path_str.c_str());
                        return false;
                    }
                }
            } else {
                struct zip_file* zip_entry = ::zip_fopen_index(m_zipfile, i, 0);
                if (!zip_entry) {
                    PrintLog("ERROR: fopen index: %d failed\n", (int)i);
                    return false;
                }
                std::FILE* fp = std::fopen(entry_path_str.c_str(), "wb+");
                if (!fp) {
                    ::zip_fclose(zip_entry);
                    PrintLog("ERROR: create local file failed %s\n", entry_path_str.c_str());
                    return false;
                }

                zip_int64_t total_size = 0;
                PrintLog("xxxx name %s %d\n", entry_path_str.c_str(), stat.size);
                while (total_size < stat.size) {
                    zip_int64_t readn = ::zip_fread(zip_entry, buf.data(), buf.size());
                    if (readn < 0) {
                        PrintLog("ERROR: zip_fread < 0 failed\n");
                        ::zip_fclose(zip_entry);
                        std::fclose(fp);
                        return false;
                    }
                    if (readn > 0) {
                        std::fwrite(buf.data(), 1, readn, fp);
                        std::fflush(fp);
                        total_size += readn;
                    }
                }
                PrintLog("xxxx2 name %s %d\n", entry_path_str.c_str(), total_size);
                ::zip_fclose(zip_entry);
                std::fflush(fp);
                std::fclose(fp);
            }
        } catch (const std::exception& e) {
            PrintLog("ERROR: exception: %s\n", e.what());
            return false;
        }
    }
    return true;
}

#undef PrintLog

