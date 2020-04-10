#include <cstdio>
#include <string>
#include <thread>
#include <chrono>
#include <filesystem>

#include "LibzipTools.h"

#include <windows.h>

static std::string GetExePath()
{
    char szFullPath[MAX_PATH];
    char szdrive[_MAX_DRIVE];
    char szdir[_MAX_DIR];
    ::GetModuleFileNameA(NULL, szFullPath, MAX_PATH);
    _splitpath_s(szFullPath, szdrive, _MAX_DRIVE, szdir, _MAX_DIR, NULL, NULL, NULL, NULL);

    std::string s1 = szdrive;
    std::string s2 = szdir;
    /*
    std::string szPath;
    szPath = StringFormat("%s%s", szdrive, szdir);
    return szPath;
    */
    return s1 + s2;
}

void Test()
{
    std::string from;
    std::string to;
    if (0) {
        from = R"(D:\vspro\test\libzip\ÖÇ»Û Íø¿§0.12 a f - ffx_f@#\500.txt)";
        to = R"(D:\vspro\test\libzip\ÖÇ»Û Íø¿§0.12 a f - ffx_f@#\x.zip)";
        //CompressWindowsFile4(from, to, "a.txt");
    }
    else {
        std::filesystem::path root_path = GetExePath();

        std::filesystem::path from_path = root_path;
        from_path /= "voice.zip";
        std::filesystem::path to_path = root_path;

        from = from_path.generic_string();
        to = to_path.generic_string();
    }

    printf("from: %s\n", from.c_str());
    printf("to: %s\n", to.c_str());

    LibZipTools uz;
    bool succ = uz.Uncompress(from, to);
    if (!succ) {
        printf("ERROR: [line:%d] uncompress2 failure ", __LINE__);
        return;
    }
    printf("INFO: [line:%d] uncompress2 success", __LINE__);
}

void TestCompress()
{
    LibZipTools tools;

    std::string from = R"(D:\nginx-1.17.0\data\dist)";
    std::string from_file = "index.html";
    std::string to = from;
    std::string to_file = "index.zip";
    bool succ = tools.CompressFile(from, from_file, to, to_file);
    printf("Compress succ:%d\n", (int)succ);
}

static void ThreadYeld()
{
    int n = 0;
    while (1) {
        std::this_thread::sleep_for(std::chrono::seconds{ 1 });
        ++n;
        //printf("sleep for: %d\n", n);
    }
}

/*
        int ecode = ::zip_error_code_zip(&error);
        SafeLog("WARNING:[%d] zip_source_win32w_create from_src failed [%d] [%s] [%s]", __LINE__, ecode, ::zip_error_strerror(&error), from_full_path.c_str());
        ::zip_error_fini(&error);
*/

void Test3()
{
    zip_error_t ze;
    ::zip_error_init(&ze);
    std::string from = R"(D:\nginx-1.17.0\data\dist\index.html)";
    zip_source_t * zs = ::zip_source_file_create(from.c_str(), 0, -1, &ze);
    if (!zs) {
        printf("zip_source_file_create failed. %s\n", from.c_str());
        return;
    }


    /*
    int n = ::zip_source_open(zs);
    if (n != 0) {
        printf("zip_source_open error %d\n", n);
        return;
    }
    */

    int ec = 0;
    std::string to = R"(D:\nginx-1.17.0\data\dist\index.zip)";
    zip_t* z = zip_open(to.c_str(), ZIP_CHECKCONS | ZIP_CREATE, &ec);
    if (!z) {
        printf("ERROR: zip_open_from_source failed. zipfile_path:[%s]\n",  to.c_str());
        return;
    }

    zip_int64_t n = ::zip_file_add(z, "a.html", zs, ZIP_FL_OVERWRITE | ZIP_FL_ENC_UTF_8);
    if (n < 0) {
        printf("ERROR: zip_file_add\n %d", (int)n);
        return;
    }

    ::zip_source_close(zs);
    ::zip_close(z);

    /*
    zip_t* z = ::zip_open_from_source(zs, ZIP_CHECKCONS | ZIP_RDONLY | ZIP_CREATE, &ze);
    if (!z) {
        printf("ERROR: zip_open_from_source failed. zipfile_path:[%s]\n",  to.c_str());
        return;
    }
    */

    printf("success %d\n", (int)n);
}

void Test4()
{
    LibZipTools tools;

    std::string from = "D:\\temp";
    std::string from_f = "a.cpp";

    std::string to = from;
    auto succ = tools.CompressFile(from, std::vector<std::string>{ "a.cpp", "b.cpp", "c.java" }, to, "bbb.zip");
    printf("success %d\n", (int)succ);
}

void Test5()
{
    LibZipTools tools;

    std::string from = "D:/nginx_test/zhwk_nginx";
    auto succ = tools.CompressDir(from, "D:/temp", "nginx.zip");
    printf("success %d\n", (int)succ);
}

int main()
{
    Test5();

    ThreadYeld();
}
