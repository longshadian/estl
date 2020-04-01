// main.cpp
#include <sqlite3.h>
#include <stdio.h>

#include <boost/asio.hpp>

int main()
{
    printf("%s\n", sqlite3_libversion());
	boost::asio::io_context ioc{};
    return 0;
}
