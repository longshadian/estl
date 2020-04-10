#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>
#include <chrono>
#include <thread>
#include <iostream>
#include <array>
#include <filesystem>
#include <cassert>
//#include <boost/filesystem.hpp>

#include <inttypes.h>
#include <zipconf.h>
#include <zip.h>

#include "Unzip.h"


#define ZIPFILE_NAME  "1.1.zip"
std::string dest = "D:\\vspro\\test\\libzip\\out\\";

namespace fs = std::filesystem;

void fun()
{
    int iErr = 0;
    struct zip * zipfile = NULL;
    struct zip_file *entries = NULL;
    struct zip_stat stat;
    zip_int64_t i64Num = 0;
    zip_int64_t i64Count = 0;
    int iRead = 0;
    int iLen = 0;
    char buf[1024];

    memset(&stat, 0, sizeof(stat));
    memset(buf, 0, sizeof(buf));

    zipfile = zip_open(ZIPFILE_NAME, ZIP_CHECKCONS, &iErr);
    if (!zipfile)
    {
        printf("zip open failed:%d\n", iErr);

        exit(EXIT_FAILURE);
    }

    //get how many entrrites in archive
    i64Num = zip_get_num_entries(zipfile, 0);

    for (i64Count = 0; i64Count < i64Num; i64Count++)
    {
        if (zip_stat_index(zipfile, i64Count, 0, &stat) == 0)
        {
            printf("the file name is:%s \t\t index: %d size: %d com_size: %d\n"
                , stat.name, stat.index, stat.size, stat.comp_size);
        }
        continue;

        entries = zip_fopen_index(zipfile, i64Count, 0);
        if (!entries)
        {
            printf("fopen index failed\n");
            goto End;
        }

        //create the original file
        std::string dest_path = dest + stat.name;
        //FILE *fp = fopen(stat.name, "w+");
        FILE *fp = fopen(dest_path.c_str(), "w+");
        if (!fp)
        {
            printf("create local file failed %s\n", dest_path.c_str());
            goto End;
        }

        while (iLen < stat.size)
        {
            iRead = zip_fread(entries, buf, 1024);
            if (iRead < 0)
            {
                printf("read file failed\n");
                fclose(fp);
                goto End;
            }

            fwrite(buf, 1, iRead, fp);
            iLen += iRead;
        }

        fclose(fp);
    }

End:
    zip_close(zipfile);

    system("pause");

    return;
}

void Fun2()
{
    const std::string root_path = "D:\\vspro\\test\\libzip\\out\\";

    int iErr = 0;
    struct zip_stat stat;
    int iRead = 0;
    int iLen = 0;
    char buf[1024];

    std::memset(&stat, 0, sizeof(stat));
    std::memset(buf, 0, sizeof(buf));

    struct zip * zipfile = zip_open(ZIPFILE_NAME, ZIP_CHECKCONS, &iErr);
    if (!zipfile) {
        printf("zip open failed:%d\n", iErr);
        exit(EXIT_FAILURE);
    }

    //get how many entrrites in archive
    zip_int64_t i64Num = zip_get_num_entries(zipfile, 0);
    for (zip_int64_t i64Count = 0; i64Count < i64Num; i64Count++) {
        if (zip_stat_index(zipfile, i64Count, 0, &stat) == 0) {
            //printf("the file name is:%s \t\t index: %d size: %d com_size: %d\n"
            //    , stat.name, stat.index, stat.size, stat.comp_size);
        }
        std::string name = stat.name;

        fs::path name_path(root_path);
        name_path = name_path / name;
        // TODO 判断目录
        if (*name.rbegin() == '/') {
            if (!fs::exists(name_path)) {
                assert(fs::create_directory(name_path));
                printf("start create dir: %s\n", name_path.generic_string().c_str());
            }
        } else {
            struct zip_file *entries = zip_fopen_index(zipfile, i64Count, 0);
            if (!entries) {
                printf("fopen index failed\n");
                goto End;
            }
            std::string file_name = name_path.generic_string();
            FILE *fp = fopen(file_name.c_str(), "w+");
            if (!fp) {
                printf("create local file failed %s\n", file_name.c_str());
                goto End;
            }

            while (iLen < stat.size) {
                iRead = zip_fread(entries, buf, 1024);
                if (iRead < 0) {
                    printf("read file failed\n");
                    fclose(fp);
                    goto End;
                }
                fwrite(buf, 1, iRead, fp);
                iLen += iRead;
            }
            fclose(fp);
        }
    }
End:
    zip_close(zipfile);
}

void Test3()
{
    std::string root_path = R"(D:\vspro\test\libzip\cef.zip)";
    std::string uncompress_path = R"(D:\vspro\test\libzip\out)";
    Unzip unzip{};
    assert(unzip.Uncompress(root_path, uncompress_path));

    /*
    std::string to = R"(D:\vspro\test\libzip\xx)";
    try {
        fs::copy_directory(fs::path(uncompress_path), fs::path(to));
    } catch (const std::exception& e) {
        printf("exception: %s\n", e.what());
    }
    */
}

void Test4()
{
    try {
        std::string from = R"(E:\InternetBar\UpdateClientTool\UpdateSvr\trunk\Debug\1.0.zip)";
        std::string to = R"(C:\Users\admin\Desktop\nginx-1.17.0\nginx-1.17.0\data\download)";
        fs::copy_file(from, to);
    } catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << "\n";
    }
}

void CopyFile(std::string from, std::string to)
{
    auto* ffrom = std::fopen(from.c_str(), "rb");
    if (!ffrom) {
        printf("fopen from %s faile.\n", from.c_str());
        return;
    }

    auto* fto = std::fopen(to.c_str(), "wb+");
    if (!fto) {
        printf("fopen to %s faile.\n", to.c_str());
        return;
    }

    std::array<char, 1024> buf{};
    while (true) {
        auto readn = std::fread(buf.data(), 1, buf.size(), ffrom);
        if (readn <= 0)
            break;

        auto writen = std::fwrite(buf.data(), 1, readn, fto);
        assert(readn == writen);
    }

    std::fflush(fto);
    std::fclose(fto);
    std::fclose(ffrom);
}

void Test5()
{
    try {
        std::string from = R"(E:\InternetBar\UpdateClientTool\UpdateSvr\trunk\Debug\1.0.zip)";
        std::string to = R"(C:\Users\admin\Desktop\nginx-1.17.0\nginx-1.17.0\data\download\1.0.zip)";
        //fs::copy_file(from, to);
        CopyFile(from, to);
    } catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << "\n";
    }
}

int main(int argc, char *argv[])
{
    Test5();
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds{ 2 });
        printf("sleep\n!");
    }
    system("pause");
    return 0;
}
