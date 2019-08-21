
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <zookeeper/zookeeper.h>
//#include <zookeeper_log.h>

#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <sstream>

bool run = true;

static void sigTerm(int v)
{
    std::cout << "xxxxx sigterm\n";
    run = false;
}

std::string GetState(int32_t state)
{
    if (state == ZOO_EXPIRED_SESSION_STATE)
         return "ZOO_EXPIRED_SESSION_STATE";
    else if (state == ZOO_AUTH_FAILED_STATE)
        return "ZOO_AUTH_FAILED_STATE";
    else if (state == ZOO_CONNECTING_STATE)
        return "ZOO_CONNECTING_STATE";
    else if (state == ZOO_ASSOCIATING_STATE)
        return "ZOO_ASSOCIATING_STATE";
    else if (state == ZOO_CONNECTED_STATE)
        return "ZOO_CONNECTED_STATE";
    return "unknown state";
}

std::string GetType(int32_t type)
{
    if (type == ZOO_CREATED_EVENT)
        return "ZOO_CREATED_EVENT";
    else if (type == ZOO_DELETED_EVENT)
        return "ZOO_DELETED_EVENT";
    else if (type == ZOO_CHANGED_EVENT)
        return "ZOO_CHANGED_EVENT";
    else if (type == ZOO_CHILD_EVENT)
        return "ZOO_CHILD_EVENT";
    else if (type == ZOO_SESSION_EVENT)
        return "ZOO_SESSION_EVENT";
    else if (type == ZOO_NOTWATCHING_EVENT)
        return "ZOO_NOTWATCHING_EVENT";
    return "unknown type";
}


void QueryServerd_watcher_global(zhandle_t * zh, int type, int state, const char *path, void *watcherCtx);
//static void QueryServerd_dump_stat(const struct Stat *stat);
void StatCompletion(int rc, const struct Stat *stat, const void *data);
void Watcher_awexists(zhandle_t *zh, int type, int state, const char *path, void *watcherCtx);

void Add_awexists(zhandle_t *zh);

void QueryServerd_watcher_global(zhandle_t * zh, int type, int state, const char *path, void *watcherCtx)
{
    if (type == ZOO_SESSION_EVENT) {
        if (state == ZOO_CONNECTED_STATE) {
            printf("Connected to zookeeper service successfully!\n");
        }
        else if (state == ZOO_EXPIRED_SESSION_STATE) {
            printf("Zookeeper session expired!\n");
        }
    }
}

void StatCompletion(int rc, const struct Stat* stat, const void *data)
{
    std::cout << __FUNCTION__ << ": " 
        << " rc: " << rc
        << " state: " << " "
        << " data: " << (const char*)data
        << "\n";
}

void StringsStatCompletion(int rc, const struct String_vector* strings, const struct Stat *stat, const void *data)
{
    std::ostringstream ostm{};
    ostm << "[";
    for (int i = 0; i != strings->count; ++i) {
        ostm << (const char*)(strings->data[i]) << " ";
    }
    ostm << "]";

    std::cout << __FUNCTION__ << ": " 
        << " rc: " << rc
        << " state: " << " "
        << " data: " << (const char*)data
        << " " << ostm.str()
        << "\n";
}

void DataCompletion(int rc, const char *value, int value_len, const struct Stat *stat, const void *data)
{
    std::cout << __FUNCTION__ << ": " 
        << " rc: " << rc
        << " state: " << " "
        << " data: " << (const char*)data
        << "\n";
}

void Watcher_awexists(zhandle_t* zh, int type, int state, const char *path, void *watcherCtx)
{
    std::cout << state << ":" << GetState(state) 
        << " " << type << ":" << GetType(type)
        << "\n";
    std::cout << "trigger watch \n";
    int n = 5;
    while (n > 0) {
        std::cout << "watch sleep " << n << "\n";
        std::this_thread::sleep_for(std::chrono::seconds{1});
        --n;
    }

    if (state == ZOO_CONNECTED_STATE) {
        if (type == ZOO_DELETED_EVENT) {
            printf("knet ZOO_DELETED_EVENT %s\n", path);
            Add_awexists(zh);
        } else if (type == ZOO_CREATED_EVENT) {
            printf("knet ZOO_CREATED_EVENT node %s\n", path);
        }
    }
    // re-exists and set watch on /QueryServer again.
    Add_awexists(zh);
}

void Add_awexists(zhandle_t* zh)
{
    std::string data = "Add_awexists";
    int ret = ZOK;
    //ret = ::zoo_awexists(zh, "/knet/mytest", Watcher_awexists, &data[0], StatCompletion, "StatCompletion");
    //ret = ::zoo_awget(zh, "/knet", Watcher_awexists, &data[0], DataCompletion, "DataCompletion");
    ret = ::zoo_awget_children2(zh, "/knet", Watcher_awexists, &data[0], StringsStatCompletion, "StringsStatCompletion");
    if (ret) {
        fprintf(stderr, "Error %d for %s\n", ret, "aexists");
        exit(EXIT_FAILURE);
    }
}

int main(int argc, const char *argv[])
{
    ::signal(SIGTERM, &::sigTerm);
    ::signal(SIGINT, &::sigTerm);


    const char* host = "127.0.0.1:2181";
    int timeout = 30000;
    std::string data = "zookeeper_init";

    ::zoo_set_debug_level(ZOO_LOG_LEVEL_WARN);
    zhandle_t *zkhandle = ::zookeeper_init(host, QueryServerd_watcher_global,
        timeout, 0, &data[0], 0);
    if (zkhandle == NULL) {
        fprintf(stderr, "Error when connecting to zookeeper servers...\n");
        exit(EXIT_FAILURE);
    }

    //zookeeper_interest(zkhandle, nullptr, nullptr, nullptr);

    Add_awexists(zkhandle);
    // Wait for asynchronous zookeeper call done.

    while (run) {
        std::this_thread::sleep_for(std::chrono::seconds{1});
        //std::cout << "sleep...\n";
    }
    ::zookeeper_close(zkhandle);

    std::cout << "main exit\n";
    return 0;
}