#include <sqlite3.h>
#include <stdio.h>

#include <spdlog/spdlog.h>
#include "net/Udp.h"

#include "TestNet.h"
#include "GlobalDefine.h"

int main()
{
    printf("%s\n", sqlite3_libversion());
	boost::asio::io_context ioc{};

    //testnet::TestNet();
    testnet::TestUdpClinet();

    return 0;
}

