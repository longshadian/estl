#include <boost/filesystem.hpp>
#include <boost/iostreams/code_converter.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include <openssl/md5.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <boost/property_tree/xml_parser.hpp>

int main()
{
    std::string path = "./md5.cpp";
    unsigned char result[MD5_DIGEST_LENGTH];
    boost::iostreams::mapped_file_source src(path);
    MD5((unsigned char*)src.data(), src.size(), result);

    std::ostringstream sout;
    sout << std::hex << std::setfill('0');
    for (long long c : result)
    {
        sout << std::setw(2) << (long long)c;
    }

    auto sign = out.str();
    std::cout << sign << "\n";


    boost::property_tree pt{};
    pt.put("xml.a", sign);
    pt.put("xml.hehe", 111);
    boost::property_tree::write_xml("x.xml", pt);

    return 0;
}

