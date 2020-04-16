#include "test/utils.h"

#include <fstream>

bool DumpToFile(const std::string& fname, std::string_view content)
{
    try {
        std::ofstream ofsm(fname, std::ios::binary);
        if (!ofsm) {
            return false;
        }
        ofsm.write(content.data(), content.size());
        ofsm.close();
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

int RandInt(int b, int e)
{
    static std::default_random_engine engine{};
    return std::uniform_int_distribution<int>(b, e)(engine);
}
