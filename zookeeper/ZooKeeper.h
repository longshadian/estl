#pragma once

#include <functional>
#include <string>

#include <zookeeper/zookeeper.h>

namespace zkcpp {

class ZooKeeper;
using WatcherCB = std::function<void(int type, int state, const char* path, ZooKeeper* zk)>;

std::string StateToString(int32_t state);
std::string TypeToString(int32_t type);

struct ZooKeeperOpt
{
    std::string m_root_path;
    std::string m_self_sid;
};

class ZooKeeper
{
public:
    ZooKeeper();
    ~ZooKeeper();
    ZooKeeper(const ZooKeeper& rhs) = delete;
    ZooKeeper& operator=(const ZooKeeper& rhs) = delete;
    ZooKeeper(ZooKeeper&& rhs) = delete;
    ZooKeeper& operator=(ZooKeeper&& rhs) = delete;

    bool                    Initialize(std::string host, const ZooKeeperOpt& opt);

    //zhandle_t*              Init(std::string host, WatcherCB cb);
    zhandle_t*              GetHandle();

private:
    static void             WatcherCB_Wrapper(zhandle_t* zk, int type, int state, const char* path, void* ctx);
    void                    Destroy();

    static void             ZKInitCallback(zhandle_t* zk, int type, int state, const char* path, void* ctx);
    static void             ZKWatchCallback(zhandle_t* zk, int type, int state, const char* path, void* ctx);
    static void             ZK_AddWatch(zhandle_t* zk);
    static void             StringsStatCompletion(int rc, const struct String_vector* strings, const struct Stat *stat, const void* data);

private:
    std::string             m_host;
    zhandle_t*              m_zk;
    WatcherCB               m_watch_cb;
    ZooKeeperOpt            m_opt;
};

} // zkcpp

