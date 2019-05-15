
#include <cassert>
#include <iostream>
#include <numeric>

#include <boost/convert.hpp>
#include <boost/convert/lexical_cast.hpp>
#include <boost/convert/stream.hpp>


/*
using std::string;
using boost::lexical_cast;
using boost::convert;
*/

#include <string>
#include <array>

struct boost::cnv::by_default : public boost::cnv::cstream {};

int main()
{
    std::array<char const*, 3> strs = { { " 5", "0XF", "not an int" } };
    std::vector<int>             ints;
    boost::cnv::cstream          cnv;

    // Configure converter to read as a string of hexadecimal characters, skip (leading) white spaces.
    cnv(std::hex)(std::skipws);

    std::transform(strs.begin(), strs.end(), std::back_inserter(ints),
        boost::cnv::apply<int>(boost::cref(cnv)).value_or(-1));

    assert(ints.size() == 3); // Number of values processed.
    assert(ints[0] == 5);    // " 5"
    assert(ints[1] == 15);    // "0XF"
    assert(ints[2] == -1);    // "not an int"
    std::string s = "a1234";
    std::cout << s << "\t" << boost::convert<int32_t>(s).value_or(-100) << "\n";

    s = "12.3";
    std::cout << s << "\t" << boost::convert<int32_t>(s).value_or(-100) << "\n";

    s = "123";
    std::cout << s << "\t" << boost::convert<int32_t>(s).value_or(-100) << "\n";

    s = "123.";
    std::cout << s << "\t" << boost::convert<int32_t>(s).value_or(-100) << "\n";

    s = " 123";
    std::cout << s << "\t" << boost::convert<int32_t>(s).value_or(-100) << "\n";

    s = "123 ";
    std::cout << s << "\t" << boost::convert<int32_t>(s).value_or(-100) << "\n";

    s = "\t123";
    std::cout << s << "\t" << boost::convert<int32_t>(s).value_or(-100) << "\n";

    s = "     123";
    std::cout << s << "\t" << boost::convert<int32_t>(s).value_or(-100) << "\n";

    s = "2147483647";
    std::cout << s << "\t" << boost::convert<int32_t>(s).value_or(-100) << "\n";

    s = "2147483648";
    std::cout << s << "\t" << boost::convert<int32_t>(s).value_or(-100) << "\n";

    s = "2147483649";
    std::cout << s << "\t" << boost::convert<int32_t>(s).value_or(-100) << "\n";

    s = "2147483650";
    std::cout << s << "\t" << boost::convert<int32_t>(s).value_or(-100) << "\n";

    std::cout << "uint32===============\n";
    s = "4294967295";
    std::cout << s << "\t" << boost::convert<uint32_t>(s).value_or(100) << "\n";

    s = "4294967296";
    std::cout << s << "\t" << boost::convert<uint32_t>(s).value_or(100) << "\n";

    s = "4294967497";
    std::cout << s << "\t" << boost::convert<uint32_t>(s).value_or(100) << "\n";

    std::cout << "int64===============\n";
    s = "2147483650";
    std::cout << s << "\t" << boost::convert<int64_t>(s).value_or(-100) << "\n";

    /*
    std::cout << "int32_t max   " << std::numeric_limits<int32_t>::max() << "\n";
    std::cout << "uint32_t max  " << std::numeric_limits<uint32_t>::max() << "\n";
    std::cout << "int64_t max   " << std::numeric_limits<int64_t>::max() << "\n";
    std::cout << "uint64_t max  " << std::numeric_limits<uint64_t>::max() << "\n";
    */

    /*
    int32_t max   2147483647
    uint32_t max  4294967295
    int64_t max   9223372036854775807
    uint64_t max  18446744073709551615
    */

    //int val = boost::convert<int>("a123").value_or(-1);
    //std::cout << val << "\n";
    //std::cout << boost::convert<int>("123").value() << "\n";
    return 0;
}
