#include <iostream>
#include <string>
#include <fstream>

#include "filesystem.h"

#include "../doctest/doctest.h"
#include "Common.h"

namespace test_readwrite_file
{
    static bool read_file(const std::string& file_name, std::string& content)
    {
        try {
            std::ifstream ifsm(file_name, std::ios::binary);
            if (!ifsm) {
                return false;
            }
            ifsm.seekg(0, std::ios::end);
            auto len = ifsm.tellg();
            ifsm.seekg(0, std::ios::beg);
            content.resize(len);
            ifsm.read(content.data(), len);
            ifsm.close();
            return true;
        } catch (const std::exception & e) {
            return false;
        }
    }

    static bool write_file(const std::string& file_name, std::string_view content)
    {
        try {
            std::ofstream ofsm(file_name, std::ios::binary | std::ios::trunc);
            if (!ofsm) {
                return false;
            }
            ofsm.write(content.data(), content.size());
            ofsm.flush();
            ofsm.close();
            return true;
        } catch (const std::exception & e) {
            return false;
        }
    }

    static bool replace_file(const std::string& old_file, const std::string& new_file)
    {
        try {
            fs::rename(old_file, new_file);
            return true;
        } catch (const std::exception & e) {
            return false;
        }
    }

} // namespace test_readwrite_file

//#define USE_TEST

#if defined (USE_TEST)
TEST_CASE("test_readwrite_file")
{
    using namespace test_readwrite_file;
    if (0) {
        std::string sold = "a.txt";
        std::string snew = "anew.txt";
        auto succ = replace_file(snew, sold);
        CHECK(succ);
    }

    if (1) {
        std::string s1; 
        s1.resize(1, 'b');
        s1 += "\r\n";
        auto succ = write_file("b.txt", s1);
        CHECK(succ);
    }
}
#endif

