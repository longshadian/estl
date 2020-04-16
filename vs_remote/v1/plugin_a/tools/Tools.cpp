#include "plugin_a/tools/Tools.h"

#include <sstream>
#include <algorithm>
#include <fstream>

namespace plugin
{
namespace a
{

bool read_file(const std::string& file_name, std::string& content, std::string& emsg)
{
    try {
        std::ifstream ifsm(file_name, std::ios::binary);
        if (!ifsm) {
            emsg = "cannot open: " + file_name;
            return false;
        }
        ifsm.seekg(0, std::ios::end);
        auto len = ifsm.tellg();
        ifsm.seekg(0, std::ios::beg);
        content.resize(len);
        ifsm.read(content.data(), len);
        ifsm.close();
        return true;
    } catch (const std::exception& e) {
        emsg = e.what();
        return false;
    }
}

} // namespace a
} // namespace plugin


