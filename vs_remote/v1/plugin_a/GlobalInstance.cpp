#include "plugin_a/GlobalInstance.h"

#include <string>
#include <fstream>

#include "plugin_a/log.h"

namespace plugin
{
namespace a
{

static GlobalInstance* g_instance = nullptr;

GlobalInstance::GlobalInstance()
{
}

GlobalInstance::~GlobalInstance()
{
}

GlobalInstance* GlobalInstance::get()
{
    if (!g_instance) {
        g_instance = new GlobalInstance();
    }
    return g_instance;
}

void GlobalInstance::cleanup()
{
    if (g_instance) {
        delete g_instance;
        g_instance = nullptr;
    }
}

std::string GlobalInstance::execute()
{
    std::string a = "aaa";
    for (int i = 0; i != 10; ++i) {
        std::string s = " " + std::to_string(i);
        std::cout << i <<"\n";
        a += s;
    }
    return a;
}

} // namespace a
} // namespace plugin
