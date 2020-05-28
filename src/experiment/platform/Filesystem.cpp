#include <iostream>
#include <filesystem>

#include "../doctest/doctest.h"
#include "Common.h"

#if defined (_MSC_VER)
 #include <windows.h>
#endif

namespace test_filesystem 
{

#if defined (_MSC_VER)
static std::string GetExePath()
{
    char szFullPath[MAX_PATH];
    char szdrive[_MAX_DRIVE];
    char szdir[_MAX_DIR];
    ::GetModuleFileNameA(NULL, szFullPath, MAX_PATH);
    _splitpath_s(szFullPath, szdrive, _MAX_DRIVE, szdir, _MAX_DIR, NULL, NULL, NULL, NULL);

    std::string szPath;
    szPath += szdrive;
    szPath += szdir;
    return szPath;
}
#endif

static std::string GetStdPath()
{
    auto p = std::filesystem::current_path();

    std::cout << "string(): " << p.string() << "\n";
    std::cout << "u8string(): " << p.u8string() << "\n";
    std::cout << "generic_string(): " << p.generic_string() << "\n";
    std::cout << "generic_u8string(): " << p.generic_u8string() << "\n";
    return p.string();
}

static int Test1()
{
#if defined (_MSC_VER)
    std::cout << "win32 GetExePath: " << GetExePath() << "\n";
#endif

    auto p = std::filesystem::current_path();
    std::cout << "string(): " << p.string() << "\n";
    std::cout << "u8string(): " << p.u8string() << "\n";
    std::cout << "generic_string(): " << p.generic_string() << "\n";
    std::cout << "generic_u8string(): " << p.generic_u8string() << "\n";


    if (1) {
        auto s = std::filesystem::space(std::filesystem::current_path());
        std::cout << "available: " << s.available /(1024*1024) << "\n";
        std::cout << "capacity: " << s.capacity /(1024*1024) << "\n";
        std::cout << "free: " << s.free/(1024 * 1024) << "\n";
    } else {
    }

    return 0;
}

static std::string file_name(std::string s)
{
    auto pos = s.rfind('/');
    if (pos == std::string::npos) {
        pos = s.rfind('\\');
    }
    if (pos == std::string::npos)
        return s;
    return s.substr(pos + 1);
}

static void TestRemove()
{
    std::string fpath = R"(D:\temp\bb\a.txt)";
    std::error_code ec;
    auto succ = std::filesystem::remove(fpath, ec);
    std::cout << (int)succ << " " << ec.value();
}

} // namespace test_filesystem

//#define USE_TEST

#if defined (USE_TEST)
TEST_CASE("TestFilesystem")
{
    LogInfo << __FILE__;
    try {
        //test_filesystem::Test1();
        test_filesystem::TestRemove();
    } catch (const std::exception& e) {
        printf("Error: exception: %s\n", e.what());
        CHECK(false);
    }
}
#endif

