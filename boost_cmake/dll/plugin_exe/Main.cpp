#include <iostream>
#include "Service.h"

#include <boost/dll/import.hpp>
#include <boost/dll.hpp>
#include <boost/function.hpp>

#define USE_WIN32API 0

#if USE_WIN32API
#include <windows.h>
#endif

typedef Export* (*Func)(Import*);
//using Func = Export* (*)(Import*);


#if USE_WIN32API
int test()
{
    HMODULE handle = LoadLibrary("plugin.dll");
    if (!handle) {
        std::cout << "load dll error\n";
        return -1;
    }
    Func func = (Func)GetProcAddress(handle, "ExportAPI");
    if (!func) {
        std::cout << "get ExportAPI error\n";
        FreeLibrary(handle);
        return -1;
    }

    Import imp;
    imp.m_version = 999;
    Export* exp = func(&imp);
    std::cout << "export version: " << exp->m_version << "\n";
    std::cout << exp->m_s->GetName() << "\n";
    std::cout << exp->m_s->Add(1,2) << "\n";
    FreeLibrary(handle);
    return 0;
}
#endif

int testBoost()
{
    std::string dll_path = "./";
    std::string dll_name = "PluginBoost";
    //boost::filesystem::path lib_path("E:\\vs_pro\\dll\\Debug\\plugin_boost.dll");
    std::cout << "Loading the plugin" << std::endl;
    try {
        /*
        auto plugin = boost::dll::import<Func>(          // type of imported symbol is located between `<` and `>`
            lib_path / "plugin_boost",
            "ExportAPI",
            boost::dll::load_mode::append_decorations              // makes `libmy_plugin_sum.so` or `my_plugin_sum.dll` from `my_plugin_sum`
            //boost::dll::load_mode::search_system_folders
            );
            */
        boost::filesystem::path lib_path(dll_path);
        boost::dll::shared_library lib(lib_path);
        if (lib.has("ExportAPI")) {
            std::cout << "has ExportAPI" << "\n";
        } else {
            std::cout << "don't has ExportAPI" << "\n";
            return 0;
        }

        Func plugin = lib.get<__declspec(dllexport)Export* (Import* imp)>("ExportAPI");
        //Func plugin = lib.get<__stdcall Func>("ExportAPI");
        Import imp;
        imp.m_version = 999;
        Export* exp = plugin(&imp);
        std::cout << "export version: " << exp->m_version << "\n";
        std::cout << exp->m_s->GetName() << "\n";
        std::cout << exp->m_s->Add(1,2) << "\n";

    } catch (std::exception e) {
        std::cout << "exception: " << e.what() << "\n";
        return -1;
    }
    return 0;
}

Export* g_export;
std::shared_ptr<boost::dll::shared_library> g_lib;
boost::function<Export*(Import*)> g_plugin;

int testBoost2()
{
    typedef Export* (Func2)(Import*);

    boost::filesystem::path lib_path("E:\\vs_pro\\dll\\Debug");
    std::cout << "Loading the plugin" << std::endl;
    try {
        auto plugin = boost::dll::import<Func2>(          // type of imported symbol is located between `<` and `>`
            lib_path / "plugin_boost",
            "ExportAPI",
            boost::dll::load_mode::append_decorations              // makes `libmy_plugin_sum.so` or `my_plugin_sum.dll` from `my_plugin_sum`
            //boost::dll::load_mode::search_system_folders
            );
        Import imp;
        imp.m_version = 999;
        Export* exp = (plugin)(&imp);
        std::cout << "export version: " << exp->m_version << "\n";
        std::cout << exp->m_s->GetName() << "\n";
        std::cout << exp->m_s->Add(1,2) << "\n";
        g_export = exp;
        g_plugin = plugin;

    } catch (std::exception e) {
        std::cout << "exception: " << e.what() << "\n";
        return -1;
    }
    return 0;
}

int testBoost3()
{
    std::string dll_path = R"(D:\cmake_builds\dll_builds\output\debug\bin)";
    std::string dll_name = "PluginBoost";

    //typedef Export* (Func3)(Import*);
    using Func3 = Export*(Import*);
    std::cout << "Loading the plugin" << std::endl;
    try {
        boost::filesystem::path lib_path(dll_path);

        g_lib = std::make_shared<boost::dll::shared_library>(lib_path / dll_name
            , boost::dll::load_mode::append_decorations);
        //boost::dll::shared_library lib("Kernel32.dll", dll::load_mode::search_system_folders);
        Func3& plugin = g_lib->get<Func3>("ExportAPI");
        Import imp;
        imp.m_version = 200;
        Export* exp = plugin(&imp);
        std::cout << "export version: " << exp->m_version << "\n";
        std::cout << "export name: " << exp->m_s->GetName() << "\n";
        std::cout << "export add: " << exp->m_s->Add(1,2) << "\n";
        g_export = exp;
        return 1;
    } catch (std::exception e) {
        std::cout << "exception: " << e.what() << "\n";
        return 0;
    }
}

int main() 
{
    if (!testBoost3())
        return 0;
    std::cout << "after export:\n";
    std::cout << "\t " << g_export->m_version << "\n";
    std::cout << "\t " << g_export->m_s->GetName() << "\n";
    std::cout << "\t " << g_export->m_s->Add(1,2) << "\n";
}
