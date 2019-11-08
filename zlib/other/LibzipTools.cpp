#include "LibzipTools.h"

#include <iostream>
#include <cstdio>
#include <ctime>
#include <array>
#include <filesystem>
#include <system_error>


// #include "Win32Transcode.h"

#define SafeLog(fmt, ...) if (1) printf("[%d] " fmt "\n", __LINE__, ##__VA_ARGS__)

/**
 * https://libzip.org/documentation/zip_source_win32w.html
 *
 * https://libzip.org/documentation/zip_source.html
 *  zip_source_t is reference counted, and created with a reference count of 1.
 *  zip_open_from_source(3), zip_file_add(3), and zip_file_replace(3) will decrement 
 *  the reference count of the zip_source_t when they are done using it, so zip_source_free(3) 
 *  only needs to be called when these functions return an error. Use zip_source_keep(3) 
 *  to increase the reference count, for example if you need the source after zip_close(3).
 */

static zip_t * WindowsZipOpen(const wchar_t* name, int flags) 
{
	zip_error_t error;
	::zip_error_init(&error);

	/* create source from buffer */
	zip_source_t* src = ::zip_source_win32w_create(name, 0, -1, &error);
	if (!src) {
		SafeLog("ERROR: zip_source_win32w_create failed [%s]", zip_error_strerror(&error));
		::zip_error_fini(&error);
		return nullptr;
	}

	/* open zip archive from source */
	zip_t* za = ::zip_open_from_source(src, flags, &error);
	if (!za) {
		SafeLog("ERROR: zip_open_from_source failed [%s]", zip_error_strerror(&error));
		::zip_source_free(src);
		::zip_error_fini(&error);
		return nullptr;
	}
	::zip_error_fini(&error);
	return za;
}

static bool SetPathLastWriteTime(const std::filesystem::path& p, std::time_t t)
{
    (void)p; (void)t;

    if (t <= 0)
        return false;
    try {
        // std::filesystem::last_write_time(p, t);
        // boost::filesystem::last_write_time(p, t);
        return true;
    } catch (const std::exception& e) {
        (void)e;
        return false;
    }
}

static bool ANSI_TO_UNICODE(const char* p, std::size_t len, std::wstring* w_out)
{
    (void)p; (void)len; (void)w_out;
    //Win32Transcode::ANSI_to_Unicode(p, len, *w_out);
    return true;
}

static bool UTF8_TO_ANSI(const char* p, std::size_t len, std::string* out)
{
    (void)p; (void)len; (void)out;
    // Win32Transcode::UTF8_to_ANSI(std::string(p, p + len));
    return true;
}

static std::string FullPath(const std::string& path, const std::string& file)
{
    try {
        std::filesystem::path p = path;
        p /= file;
        return p.generic_string();
    } catch (const std::exception& e) {
        (void)e;
        return "";
    }
}

static bool ZipEntryIsDir(const std::string& entry_name)
{
    // TODO is directory??
    if (entry_name.empty())
        return false;
    if (*entry_name.rbegin() == '/')
        return true;
    return false;
}

LibZipTools::LibZipTools()
    : m_zipfile(nullptr)
{
}

LibZipTools::~LibZipTools()
{
    if (m_zipfile) {
        ::zip_close(m_zipfile);
        m_zipfile = nullptr;
    }
}

void LibZipTools::Reset()
{
    if (m_zipfile) {
        ::zip_close(m_zipfile);
        m_zipfile = nullptr;
    }
}

bool LibZipTools::Uncompress(std::string zipfile_path, std::string uncompress_path)
{
	if (0) {
		int error_code = 0;
		m_zipfile = ::zip_open(zipfile_path.c_str(), ZIP_CHECKCONS| ZIP_RDONLY, &error_code);
		if (!m_zipfile) {
			SafeLog("ERROR: zip open failed:[%d] zipfile_path:[%s]\n", error_code, zipfile_path.c_str());
			return false;
		}
	} else {
		// zipfile_path可能包含中文路径，这里需要转成unicode
		std::wstring zipfile_path_w; 
        ANSI_TO_UNICODE(zipfile_path.c_str(), zipfile_path.length(), &zipfile_path_w);
		m_zipfile = WindowsZipOpen(zipfile_path_w.c_str(), ZIP_CHECKCONS | ZIP_RDONLY);
		if (!m_zipfile) {
			SafeLog("ERROR: zip open failed zipfile_path:[%s] path_len:[%d] wpath_len:[%d]"
				, zipfile_path.c_str(), (int)zipfile_path.size(), (int)zipfile_path_w.size());
			return false;
		}
	}

    std::array<char, 1024* 10> buf{};
    struct zip_stat stat;
    zip_int64_t num_enties = ::zip_get_num_entries(m_zipfile, 0);
    for (zip_int64_t i = 0; i < num_enties; i++) {
        std::memset(&stat, 0, sizeof(stat));
        if (::zip_stat_index(m_zipfile, i, 0, &stat) == 0) {
            //SafeLog("file name is:%s \t\t index: %d size: %d com_size: %d\n", stat.name, stat.index, stat.size, stat.comp_size);
        }

        std::string entry_path_str;
        std::string entry_name = stat.name;
        std::time_t last_wt = stat.mtime;

        std::string entry_name_ansi;
        UTF8_TO_ANSI(entry_name.data(), entry_name.length(), &entry_name_ansi);

        std::filesystem::path entry_path;
        if (ZipEntryIsDir(entry_name_ansi)) {
            try {
                entry_path = uncompress_path;
                entry_path = entry_path / entry_name_ansi;
                entry_path_str = entry_path.generic_string();
                if (!std::filesystem::exists(entry_path)) {
                    if (!std::filesystem::create_directory(entry_path)) {
                        SafeLog("ERROR: create dir: %s failed.\n", entry_path_str.c_str());
                        return false;
                    }
                    SafeLog("INFO: create dir: %s success.\n", entry_path_str.c_str());
                }
                SetPathLastWriteTime(entry_path, last_wt);
            } catch (const std::exception& e) {
                SafeLog("ERROR: exception: %s\n", e.what());
                return false;
            }
            continue;
        }

        struct zip_file* zip_entry = ::zip_fopen_index(m_zipfile, i, 0);
        if (!zip_entry) {
            SafeLog("ERROR: fopen index: %d failed\n", (int)i);
            return false;
        }

        std::FILE* fp = std::fopen(entry_path_str.c_str(), "wb+");
        if (!fp) {
            ::zip_fclose(zip_entry);
            SafeLog("ERROR: create local file failed %s\n", entry_path_str.c_str());
            return false;
        }

        zip_int64_t total_size = 0;
        SafeLog("INFO: start write file: %s   size %llu %llu\n", entry_path_str.c_str(), static_cast<std::uint64_t>(stat.size), static_cast<std::uint64_t>(stat.comp_size));
        while (total_size < static_cast<zip_int64_t>(stat.size)) {
            zip_int64_t readn = ::zip_fread(zip_entry, buf.data(), buf.size());
            if (readn < 0) {
                SafeLog("ERROR: zip_fread < 0 failed\n");
                std::fclose(fp);
                ::zip_fclose(zip_entry);
                return false;
            }
            std::fwrite(buf.data(), 1, static_cast<std::size_t>(readn), fp);
            std::fflush(fp);
            total_size += readn;
        }
        std::fflush(fp);
        std::fclose(fp);
        ::zip_fclose(zip_entry);
        SetPathLastWriteTime(entry_path, last_wt);
    }
    return true;
}

/*
bool LibZipTools::CompressWindowsFile(const std::string& from_path, const std::string& from_filename, const std::string& to_path, const std::string& zip_name)
{
    std::wstring to_path_w;
    ANSI_TO_UNICODE(to_path.c_str(), to_path.length(), &to_path_w);
    {
        zip_t* to_zip = WindowsZipOpen(to_path_w.c_str(), ZIP_CHECKCONS | ZIP_CREATE);
        if (!to_zip) {
            SafeLog("WARNING:[%d] WindowsZipOpen to_zip failed. [%s]", __LINE__, to_path.c_str());
            return false;
        }
        m_zipfile = to_zip;
    }

    std::wstring from_path_w;
    ANSI_TO_UNICODE(from_path.c_str(), from_path.length(), &from_path_w);
    zip_error_t error;
    ::zip_error_init(&error);
    zip_source_t* from_src = ::zip_source_win32w_create(from_path_w.c_str(), 0, -1, &error);
    if (!from_src) {
        int ecode = ::zip_error_code_zip(&error);
        SafeLog("WARNING:[%d] zip_source_win32w_create from_src failed [%d] [%s] [%s]", __LINE__, ecode, ::zip_error_strerror(&error), from_path.c_str());
        ::zip_error_fini(&error);
        return false;
    }

    bool succ = false;
    zip_int64_t ret = ::zip_file_add(m_zipfile, fname.c_str(), from_src, ZIP_FL_OVERWRITE);
    if (ret < 0) {
        SafeLog("WARNING:[%d] zip_file_add failed ret:[%d] [%s]", __LINE__, static_cast<int>(ret), fname.c_str());
        succ = false;
    }
    ::zip_error_fini(&error);
    //::zip_source_free(from_src);
    succ = true;
    return succ;
}
*/

bool LibZipTools::CompressWindowsFile2(const std::string& from_path, const std::string& from_file, const std::string& to_path, const std::string& to_file)
{
    std::string to_full_path = FullPath(to_path, to_file);
    std::wstring to_full_path_w;
    ANSI_TO_UNICODE(to_full_path.c_str(), to_full_path.length(), &to_full_path_w);
    {
        zip_t* to_zip = WindowsZipOpen(to_full_path_w.c_str(), ZIP_CHECKCONS | ZIP_CREATE);
        if (!to_zip) {
            SafeLog("WARNING:[%d] WindowsZipOpen to_zip failed. [%s]", __LINE__, to_full_path.c_str());
            return false;
        }
        m_zipfile = to_zip;
    }

    std::string from_full_path = FullPath(from_path, from_file);
    std::wstring from_full_path_w;
    ANSI_TO_UNICODE(from_full_path.c_str(), from_full_path.length(), &from_full_path_w);
    zip_error_t error;
    ::zip_error_init(&error);
    zip_source_t* from_src = ::zip_source_win32w_create(from_full_path_w.c_str(), 0, -1, &error);
    if (!from_src) {
        int ecode = ::zip_error_code_zip(&error);
        SafeLog("WARNING:[%d] zip_source_win32w_create from_src failed [%d] [%s] [%s]", __LINE__, ecode, ::zip_error_strerror(&error), from_full_path.c_str());
        ::zip_error_fini(&error);
        return false;
    }

    bool succ = false;
    zip_int64_t ret = ::zip_file_add(m_zipfile, from_file.c_str(), from_src, ZIP_FL_OVERWRITE);
    if (ret < 0) {
        SafeLog("WARNING:[%d] zip_file_add failed ret:[%d] [%s]", __LINE__, static_cast<int>(ret), to_full_path.c_str());
        succ = false;
    }
    ::zip_error_fini(&error);
    //::zip_source_free(from_src);
    succ = true;
    return succ;
}

bool LibZipTools::CompressFile(const std::string& from_path, const std::string& from_file, const std::string& to_path, const std::string& to_file)
{
    std::string from_full_path = FullPath(from_path, from_file);
    if (from_full_path.empty())
        return false;

    zip_error_t z_error;
    ::zip_error_init(&z_error);
    zip_source_t* z_s = ::zip_source_file_create(from_full_path.c_str(), 0, -1, &z_error);
    if (!z_s) {
        SafeLog("WARNING: zip_source_file_create failed. f: %s emsg: %s", from_full_path.c_str(), ::zip_error_strerror(&z_error));
		::zip_error_fini(&z_error);
        return false;
    }

    int ecode = 0;
    std::string to_full_path = FullPath(to_path, to_file);
    zip_t* z = ::zip_open(to_full_path.c_str(), ZIP_CHECKCONS | ZIP_CREATE, &ecode);
    if (!z) {
        SafeLog("WARNING: zip_open failed. f: %s ecode: %d", to_full_path.c_str(), ecode);
        ::zip_source_free(z_s);
		::zip_error_fini(&z_error);
        return false;
    }

    zip_int64_t n = ::zip_file_add(z, from_file.c_str(), z_s, ZIP_FL_OVERWRITE | ZIP_FL_ENC_UTF_8);
    if (n < 0) {
        SafeLog("WARNING: zip_file_add failed. %d", (int)n);
        ::zip_close(z);
        ::zip_source_free(z_s);
		::zip_error_fini(&z_error);
        return false;
    }

    ::zip_close(z);
    ::zip_source_free(z_s);
    ::zip_error_fini(&z_error);
    return true;
}

bool LibZipTools::CompressFile(const std::string& from_path, const std::vector<std::string>& file_list, const std::string& to_path, const std::string& to_file)
{
    if (file_list.empty())
        return false;

    int ecode = 0;
    std::string to_full_path = FullPath(to_path, to_file);
    zip_t* z = ::zip_open(to_full_path.c_str(), ZIP_CHECKCONS | ZIP_CREATE, &ecode);
    if (!z) {
        SafeLog("WARNING: zip_open failed. f: %s ecode: %d", to_full_path.c_str(), ecode);
        return false;
    }

    zip_error_t z_error;
    ::zip_error_init(&z_error);
    for (const std::string& fname : file_list) {
        std::string from_full_path = FullPath(from_path, fname);
        if (from_full_path.empty()) {
            ::zip_close(z);
            return false;
        }

        zip_source_t* z_source = ::zip_source_file_create(from_full_path.c_str(), 0, -1, &z_error);
        if (!z_source) {
            SafeLog("WARNING: zip_source_file_create failed. f: %s emsg: %s", from_full_path.c_str(), ::zip_error_strerror(&z_error));
            ::zip_error_fini(&z_error);
            ::zip_close(z);
            return false;
        }

        zip_int64_t n = ::zip_file_add(z, fname.c_str(), z_source, ZIP_FL_OVERWRITE | ZIP_FL_ENC_UTF_8);
        if (n < 0) {
            SafeLog("WARNING: zip_file_add failed. %d", (int)n);
            ::zip_source_free(z_source);
            ::zip_error_fini(&z_error);
            ::zip_close(z);
            return false;
        }
        //::zip_source_free(z_source);
    }
    ::zip_error_fini(&z_error);
    ::zip_close(z);
    return true;
}

bool LibZipTools::CompressDir(const std::string& from_path, const std::string& to_path, const std::string& to_file)
{
    std::error_code ec;
    if (!std::filesystem::exists(from_path, ec))
        return false;

    std::filesystem::path p_from_path(from_path);
    if (!std::filesystem::is_directory(p_from_path, ec)) {
        return false;
    }

    // 创建zip
    int ecode = 0;
    std::string to_full_path = FullPath(to_path, to_file);
    zip_t* z = ::zip_open(to_full_path.c_str(), ZIP_CHECKCONS | ZIP_CREATE, &ecode);
    if (!z) {
        SafeLog("WARNING: zip_open failed. f: %s ecode: %d", to_full_path.c_str(), ecode);
        return false;
    }

    zip_error_t z_error;
    ::zip_error_init(&z_error);

    std::filesystem::directory_entry ety{ p_from_path };
    std::filesystem::recursive_directory_iterator di_list{ ety };
    for (const auto& entry : di_list) {
        auto entry_type = entry.status().type();
        auto entry_path = entry.path();
        if (entry_type == std::filesystem::file_type::directory) {
            std::string dir_relative_name = std::filesystem::relative(entry_path, p_from_path, ec).generic_string();
            std::cout << "dir: " << dir_relative_name << "\n";

            if (ec) {
                SafeLog("WARNING: dir relative failed. code: %d  dir_name: %s", ec.value(),  dir_relative_name.c_str());
                ::zip_error_fini(&z_error);
                ::zip_close(z);
                return false;
            }

            zip_int64_t n = ::zip_dir_add(z, dir_relative_name.c_str(), ZIP_FL_ENC_UTF_8);
            if (n < 0) {
                SafeLog("WARNING: zip_dir_add failed. code: %d  dir_name: %s", (int)n, dir_relative_name.c_str());
                ::zip_error_fini(&z_error);
                ::zip_close(z);
                return false;
            }

            /*
            std::cout << "isdir:\n";
            std::cout << "\t\t" << "relative:             " << std::filesystem::relative(p_path, p_from_path).generic_string() << "\n";
            std::cout << "\t\t" << "path:                 " << p_path << "\n";
            std::cout << "\t\t" << "root_name:            " << p_path.root_name() << "\n";
            std::cout << "\t\t" << "root_directory:       " << p_path.root_directory() << "\n";
            std::cout << "\t\t" << "root_path:            " << p_path.root_path() << "\n";
            std::cout << "\t\t" << "relative_path:        " << p_path.relative_path() << "\n";
            std::cout << "\t\t" << "parent_path:          " << p_path.parent_path() << "\n";
            std::cout << "\t\t" << "filename:             " << p_path.filename() << "\n";
            std::cout << "\t\t" << "stem:                 " << p_path.stem() << "\n";
            std::cout << "\t\t" << "extension:            " << p_path.extension() << "\n";
            */

        } else if (entry_type == std::filesystem::file_type::regular) {
            std::string file_relative_name = std::filesystem::relative(entry_path, p_from_path, ec).generic_string();
            std::string file_full_name = entry_path.generic_string();


            std::cout << "file: " << file_relative_name << "\n";

            zip_source_t* z_source = ::zip_source_file_create(file_full_name.c_str(), 0, -1, &z_error);
            if (!z_source) {
                SafeLog("WARNING: zip_source_file_create failed. f: %s emsg: %s", file_full_name.c_str(), ::zip_error_strerror(&z_error));
                ::zip_error_fini(&z_error);
                ::zip_close(z);
                return false;
            }

            zip_int64_t n = ::zip_file_add(z, file_relative_name.c_str(), z_source, ZIP_FL_OVERWRITE | ZIP_FL_ENC_UTF_8);
            if (n < 0) {
                SafeLog("WARNING: zip_file_add failed. %d", (int)n);
                ::zip_source_free(z_source);
                ::zip_error_fini(&z_error);
                ::zip_close(z);
                return false;
            }


            /*
            std::cout << "isfile:\n";
            std::cout << "\t" << "relative:             " << std::filesystem::relative(p_path, p_from_path).generic_string() << "\n";
            std::cout << "\t" << "path:                 " << p_path << "\n";
            std::cout << "\t" << "root_name:            " << p_path.root_name() << "\n";
            std::cout << "\t" << "root_directory:       " << p_path.root_directory() << "\n";
            std::cout << "\t" << "root_path:            " << p_path.root_path() << "\n";
            std::cout << "\t" << "relative_path:        " << p_path.relative_path() << "\n";
            std::cout << "\t" << "parent_path:          " << p_path.parent_path() << "\n";
            std::cout << "\t" << "filename:             " << p_path.filename() << "\n";
            std::cout << "\t" << "stem:                 " << p_path.stem() << "\n";
            std::cout << "\t" << "extension:            " << p_path.extension() << "\n";
            */

            //SafeLog("is file: %s", p_full_path.c_str());
        }
    }

    ::zip_error_fini(&z_error);
    ::zip_close(z);
    return true;
}

