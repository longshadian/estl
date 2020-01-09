#include <iostream>
#include <filesystem>

#if defined (WIN32)
 #include <windows.h>
#endif

namespace test_filesystem 
{

#if defined (WIN32)
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

static void Test1()
{
#if defined (WIN32)
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
}

static void TestRemove()
{
    std::string fpath = R"(D:\temp\bb\a.txt)";
    std::error_code ec;
    auto succ = std::filesystem::remove(fpath, ec);
    std::cout << (int)succ << " " << ec.value();
}

} // namespace test_filesystem

int TestFilesystem()
{
    try {
        //test_filesystem::Test1();
        test_filesystem::TestRemove();
        return 1;
    } catch (const std::exception& e) {
        printf("Error: exception: %s\n", e.what());
        return 0;
    }
}

