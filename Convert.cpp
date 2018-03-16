#include <iostream>

#include <cstdlib>
#include <cstring>
#include <limits>

#include <boost/convert.hpp>
#include <boost/convert/stream.hpp>

#include "SafeString.h"

//struct boost::cnv::by_default : public boost::cnv::cstream {};

void numericLimits()
{
    std::cout << "char max:               " << (int)std::numeric_limits<char>::max() << "\n";
    std::cout << "unsigned char max       " << (int)std::numeric_limits<unsigned char>::max() << "\n";
    std::cout << "short max:              " << std::numeric_limits<short>::max() << "\n";
    std::cout << "unsigned short max      " << std::numeric_limits<unsigned short>::max() << "\n";
    std::cout << "int max:                " << std::numeric_limits<int>::max() << "\n";
    std::cout << "unsigned int max:       " << std::numeric_limits<unsigned int>::max() << "\n";
    std::cout << "long max:               " << std::numeric_limits<long>::max() << "\n";
    std::cout << "long long max:          " << std::numeric_limits<long long>::max() << "\n";
    std::cout << "unsigned long max:      " << std::numeric_limits<unsigned long>::max() << "\n";
    std::cout << "unsigned long long max: " << std::numeric_limits<unsigned long long>::max() << "\n";
}

template <typename T>
struct ConvertEx
{
    static_assert(std::is_arithmetic<T>::value && !std::is_same<bool, T>::value
        , "T must be arithmetic and can't be bool!");

    static T cvt_noexcept(const char* p)
    {
        try {
            return boost::convert<T>(p, boost::cnv::cstream{}).value();
        } catch (std::exception e) {
            //std::cout << "boost convert exception:" << e.what() << "\n";
            return T{};
        }
    }

    static T cvt_noexcept(const std::string& p)
    {
        try {
            return boost::convert<T>(p, boost::cnv::cstream{}).value();
        } catch (std::exception e) {
            //std::cout << "boost convert exception:" << e.what() << "\n";
            return T{};
        }
    }

    static T cvt(const char* p)
    {
        return boost::convert<T>(p, boost::cnv::cstream{}).value();
    }

    static T cvt(const std::string& p)
    {
        return boost::convert<T>(p, boost::cnv::cstream{}).value();
    }
};

/*
template <typename T>
T cvt(const char* p)
{
    static_assert(std::is_arithmetic<T>::value && !std::is_same<bool, T>::value
        , "T must be arithmetic and can't be bool!");

    try {
        return boost::convert<T>(p, boost::cnv::cstream{}).value();
    } catch (std::exception e) {
        std::cout << "boost convert exception:" << e.what() << "\n";
        return T{};
    }
}
*/

std::vector<uint8_t> str(const char* p)
{
    auto len = std::strlen(p);
    if (len == 0)
        return {};
    std::vector<uint8_t> buf{};
    buf.resize(len);
    std::memcpy(buf.data(), p, len);
    return buf;
}

int boostConvert()
{
    try {
        auto buf = str("12345678");
        std::array<uint8_t, 2> arr = { '2', '3' };
        (void)arr;
        std::cout << ConvertEx<int>::cvt((const char*)buf.data()) << "\n";
    } catch (std::exception e) {
        std::cout << "exception " << e.what() << "\n";
    }

    return 0;
}

/*
char max :                  127
unsigned char max           255
short max :                 32767
unsigned short max          65535
int max :                   2147483647
unsigned int max :          4294967295
long max :                  9223372036854775807
long long max :             9223372036854775807
unsigned long max :         18446744073709551615
unsigned long long max :    18446744073709551615
*/

void printSafeString(const mysqlcpp::SafeString& sf)
{
    std::cout << "capacity " << sf.getCapacity() << "\n";
    std::cout << "length   " << sf.getLength() << "\n";
    std::cout << "bin size " << sf.getBinary().size() << "\n";
    std::cout << "cstring  " << sf.getCString() << "\n";
    std::cout << "===================\n";
}

int main()
{
    //numericLimits();

    //std::cout << "===================\n";
    //myConvert();
    boostConvert();

    std::cout << "bool      " << std::is_arithmetic<bool>::value << "\n";
    std::cout << "char      " << std::is_arithmetic<char>::value << "\n";
    std::cout << "short     " << std::is_arithmetic<short>::value << "\n";
    std::cout << "int       " << std::is_arithmetic<int>::value << "\n";
    std::cout << "long      " << std::is_arithmetic<long>::value << "\n";
    std::cout << "float     " << std::is_arithmetic<float>::value << "\n";
    std::cout << "double    " << std::is_arithmetic<double>::value << "\n";
    std::cout << "int*      " << std::is_arithmetic<int*>::value << "\n";
    std::cout << "nullptr_t " << std::is_arithmetic<nullptr_t>::value << "\n";
    std::cout << "===================\n";

    std::string s = "12345678";

    ::mysqlcpp::SafeString sf{};
    printSafeString(sf);

    sf.resize(s.size());
    std::memcpy(sf.getPtr(), s.c_str(), s.size());
    printSafeString(sf);

    {
        std::cout << "start move\n";
        ::mysqlcpp::SafeString sf_ex = std::move(sf);
        printSafeString(sf);
        printSafeString(sf_ex);
    }

    {
        std::cout << "start copy\n";
        sf.clear();
        sf.resize(s.size());
        std::memcpy(sf.getPtr(), s.c_str(), s.size());
        printSafeString(sf);

        ::mysqlcpp::SafeString sf_ex = sf;
        printSafeString(sf);
        printSafeString(sf_ex);
        std::cout << "after move\n";
    }


    return 0;
}
