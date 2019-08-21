#include <stdio.h>
#include <string.h>
#include <signal.h>

#include <time.h>
#include <errno.h>
#include <assert.h>

#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <sstream>

#include "ZooKeeper.h"

bool run = true;

static void sigTerm(int v)
{
    std::cout << "xxxxx sigterm\n";
    run = false;
}

void QueryServer_watcher_g(zhandle_t* zh, int type, int state, const char* path, void* WATCHER_CTX)
{
    if (type == ZOO_SESSION_EVENT) {
        if (state == ZOO_CONNECTED_STATE) {
            std::cout << "connected to zookeeper service successfully " << std::this_thread::get_id() << "\n";
        } else if (state == ZOO_EXPIRED_SESSION_STATE) {
            std::cout << "connected to zookeeper service expired " << std::this_thread::get_id() << "\n";
        }
    }
}

void QueryServer_string_completion(int rc, const char* name, const void* data)
{
    std::ostringstream ostm{};
    ostm << __FUNCTION__ << ":";
    ostm << "  name: ";
    if (name)
        ostm << name;
    else 
        ostm << "null";
    ostm << "  data: ";
    if (data)
        ostm << (const char*)data;
    else
        ostm << "null";
    if (!rc) {
        fprintf(stderr, "\tname = %s\n", name);
    }

    std::cout << ostm.str() << "  " << std::this_thread::get_id() << "\n";
}

int main(int argc, char **argv) 
{
    ::signal(SIGTERM, sigTerm);
    ::signal(SIGINT, sigTerm);

    ::zoo_set_debug_level(ZOO_LOG_LEVEL_WARN);
    //::zoo_deterministic_conn_order(1); // enable deterministic order
    std::string hostPort = "127.0.0.1:2181";
    int32_t timeout = 30000;
    std::string my_context = "xxxxxxxxa";

    zhandle_t* zh = ::zookeeper_init(hostPort.c_str(), QueryServer_watcher_g, timeout, nullptr
        , (void*)my_context.data(), 0);
    if (!zh) {
        std::cout << "ERROR: zookeeper_init \n";
        return errno;
    }

    int ret = ::zoo_acreate(zh, "/knet/mytest2", "alive", 5,
        &ZOO_OPEN_ACL_UNSAFE, ZOO_EPHEMERAL,
        QueryServer_string_completion, "zoo_acreate data");
    if (ret) {
        fprintf(stderr, "Error %d for %s\n", ret, "acreate");
        exit(EXIT_FAILURE);
    }

    do {
        // 模拟 QueryServer 对外提供服务.
        // 为了简单起见, 我们在此调用一个简单的函数来模拟 QueryServer.
        // 然后休眠 5 秒，程序主动退出(即假设此时已经崩溃).
        std::cout << "sleep " << std::this_thread::get_id() << "\n";
        std::this_thread::sleep_for(std::chrono::seconds{ 5 });
    } while (run);

    ::zookeeper_close(zh);

    std::cout << "main exit\n";
    return 0;
}
