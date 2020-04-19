#include <memory>

#if defined (__GNUC__)
    #if __GNUC__ < 8
        #include <experimental/filesystem>
        namespace fs = std::experimental::filesystem;
    #else
        #include <filesystem>
        namespace fs = std::filesystem;
    #endif
#else
    #include <filesystem>
    namespace fs = std::filesystem;
#endif

#include "test/plugin.h"

#if defined (_MSC_VER)
static const std::string g_so_name = "./a.dll";
#else
static const std::string g_so_name = "./a.so";
#endif

std::string GetPluginPath()
{
    auto p = fs::current_path();
    p /= g_so_name;
    std::string s = p.generic_string();
    //std::cout << s << "\n";
    return s;
}

std::shared_ptr<BoostPlugin> CreatePlugin()
{
    return std::make_shared<BoostPlugin>(GetPluginPath());
}

int Test();

void TestUtf8()
{
    system("chcp 65001");
    //int i = "aaa";
    std::string a = "发发发 ";
    std::cout << a << "\n";
}


int main(int argc, char** argv) {
    (void)argc; (void)argv;
    TestUtf8();

    LOG_INFO << "plugin path: " << GetPluginPath();
    int i = 0;
    ++i;
    Test();
    return EXIT_SUCCESS;
}
