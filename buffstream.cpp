#include <cstring>

#include <string>
#include <iostream>

#include <boost/asio.hpp>

void printBuf(const boost::asio::streambuf& buf)
{
    std::cout << "size:" << buf.size() << "\n";
    std::cout << "max_size:" << buf.max_size() << "\n";
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