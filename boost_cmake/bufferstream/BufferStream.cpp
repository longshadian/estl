#include <cstring>

#include <string>
#include <iostream>

#include <boost/asio.hpp>

void printBuf(const boost::asio::streambuf& buf)
{
    std::cout << "size:" << buf.size() << "\n";
    std::cout << "max_size:" << buf.max_size() << "\n";
}

void test()
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

	std::array<char, 1000> all{ 0 };
	std::cout << "get:" << b.sgetn(all.data(), all.max_size()) << "\n";
	std::cout << all.data() << "\n";
	std::cout << b.size() << "\n";
}

int main()
{
    boost::asio::streambuf buf{};
    buf.prepare(500);

    printBuf(buf);

    int32_t v = 123456;
    std::ostream ostm{ &buf };

    //ostm << v;
    //ostm.write(&v, sizeof(v));
    ostm.write((const char*)&v, (int)sizeof(v));

    int32_t x = 0;
    std::istream istm{ &buf };
    istm.read((char*)&x, (int)sizeof(x));
    std::cout << "read value " << x << "\n";

    printBuf(buf);

    return 0;
}