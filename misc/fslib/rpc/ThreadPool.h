#pragma once

#include <deque>
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <condition_variable>

namespace fslib {
namespace grpc {

class RpcRequest;

class ThreadPool
{
public:
                        ThreadPool(size_t n = 1);
                        ~ThreadPool();

    void                submit(std::unique_ptr<RpcRequest> request);
    void                stop();
private:
    void                        run();
    std::unique_ptr<RpcRequest> waitAndPop();
    std::unique_ptr<RpcRequest> tryPop();
    bool                        empty() const;
private:
    typedef std::deque<std::unique_ptr<RpcRequest>> RpcMessageQueue;
    RpcMessageQueue                                 m_messages;
    std::vector<std::thread>                        m_threads;             
    std::atomic<bool>                               m_run = { false };
    mutable std::mutex                              m_mtx;
    std::condition_variable                         m_cond;
};

}
}
