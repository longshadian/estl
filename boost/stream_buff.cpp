
#include <iostream>
#include <string>
#include <array>
#include <type_traits>

#include <boost/asio.hpp>
#include <boost/format.hpp>

int main()
{
    std::string s = "Hello, World!";

    boost::asio::streambuf b{};
    std::ostream os(&b);
    os << s;

    //std::cout << b.data() << "\n";
    std::cout << s.size() << " " << b.size() << "\n";

    b.consume(2);

    std::string p = "22231 3";
    b.sputn(p.c_str(), p.length());

    std::string ss;
    std::istream is(&b);
    is >> ss;
    std::cout << ss << "\n";
    std::cout << ss.size() << " " << b.size() << "\n";

    std::array<char, 1000> all{0};
    std::cout << "get:" << b.sgetn(all.data(), all.max_size()) << "\n";
    std::cout << all.data() << "\n";
    std::cout << b.size() << "\n";

    /*
    std::cout << std::is_same<const char*, boost::asio::streambuf::const_buffers_type>::value << "\n";
    std::cout << std::is_same<uint8_t, boost::asio::streambuf::const_buffers_type>::value << "\n";
    std::cout << std::is_same<uint8_t*, boost::asio::streambuf::const_buffers_type>::value << "\n";
    std::cout << std::is_same<const uint8_t*, boost::asio::streambuf::const_buffers_type>::value << "\n";

    std::cout << std::is_same<int8_t, boost::asio::streambuf::const_buffers_type>::value << "\n";
    std::cout << std::is_same<int8_t*, boost::asio::streambuf::const_buffers_type>::value << "\n";
    std::cout << std::is_same<const int8_t*, boost::asio::streambuf::const_buffers_type>::value << "\n";
    */

    return 0;
}
