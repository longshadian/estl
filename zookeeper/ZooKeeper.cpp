#include "ZooKeeper.h"

#include <iostream>

#define LOG(level) std::cout << #level

namespace zkcpp {
std::string StateToString(int32_t state)
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

std::string TypeToString(int32_t type)
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


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
ZooKeeper::ZooKeeper()
    : m_host()
    , m_zk()
    , m_watch_cb()
{
}

ZooKeeper::~ZooKeeper()
{
    if (m_zk)
        Destroy();
}

bool ZooKeeper::Initialize(std::string host, const ZooKeeperOpt& opt)
{
    if (m_zk) {
        LOG(WARNING) << "zk is connected. can't Initialize!";
        return false;
    }
    m_host = std::move(host);
    m_opt = opt;
    m_zk = ::zookeeper_init(m_host.c_str(), ZooKeeper::ZKInitCallback, 30000, nullptr, nullptr, 0);
    if (!m_zk) {
        LOG(ERROR) << "zookeeper_init failed. errno: " << errno 
            << " host: " << m_host;
        return false;
    }

    int32_t ret = ZOK;
    ret = ::zoo_awget_children2(m_zk, m_opt.m_root_path.c_str(), Watcher_awexists, &data[0], StringsStatCompletion, "StringsStatCompletion");
    if (ret) {
        fprintf(stderr, "Error %d for %s\n", ret, "aexists");
        exit(EXIT_FAILURE);
    }

    return true;
}

/*
zhandle_t* ZooKeeper::Init(std::string host, WatcherCB cb)
{
    int32_t timeout = 30000;
    if (m_zk)
        Destroy();
    m_host = std::move(host);
    m_zk = ::zookeeper_init(m_host.c_str(), &ZooKeeper::WatcherCB_Wrapper, timeout, nullptr, this, 0);
    return GetZHandle();
}
*/

zhandle_t* ZooKeeper::GetHandle()
{
    return m_zk;
}

void ZooKeeper::WatcherCB_Wrapper(zhandle_t* zk, int type, int state, const char* path, void* ctx)
{
    ZooKeeper* p_this = reinterpret_cast<ZooKeeper*>(ctx);
    p_this->m_watch_cb(type, state, path, p_this);
}

void ZooKeeper::Destroy()
{
    ::zookeeper_close(m_zk);
    m_zk = nullptr;
}

void ZooKeeper::ZKInitCallback(zhandle_t* zk, int type, int state, const char* path, void* ctx)
{
    LOG(INFO) << " type: " << TypeToString(type) << " state: " << StateToString(state);
    /*
    if (type == ZOO_SESSION_EVENT) {
        if (state == ZOO_CONNECTED_STATE) {
            printf("Connected to zookeeper service successfully!\n");
        } else if (state == ZOO_EXPIRED_SESSION_STATE) {
            printf("Zookeeper session expired!\n");
        }
    }
    */
}

void ZooKeeper::ZKWatchCallback(zhandle_t* zk, int type, int state, const char* path, void* ctx)
{
    LOG(INFO) << " type: " << TypeToString(type) << " state: " << StateToString(state);
    ZK_AddWatch(zk);
}

void ZooKeeper::ZK_AddWatch(zhandle_t* zk)
{
    std::string data = "Add_awexists";
    int32_t ret = ZOK;
    ret = ::zoo_awget_children2(zk, "/knet", Watcher_awexists, &data[0], StringsStatCompletion, "StringsStatCompletion");
    if (ret) {
        fprintf(stderr, "Error %d for %s\n", ret, "aexists");
        exit(EXIT_FAILURE);
    }
}

void ZooKeeper::StringsStatCompletion(int rc, const struct String_vector* strings, const struct Stat *stat, const void *data)
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


} // zkcpp
