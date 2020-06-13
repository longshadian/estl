#include "ThreadPool.h"

#include "RpcRequest.h"

namespace fslib {
namespace grpc {

ThreadPool::ThreadPool(size_t n)
{
    m_run.store(true);
    for (size_t i = 0; i != n; ++i) {
        m_threads.push_back(std::thread(&ThreadPool::run, this));
    }
}

ThreadPool::~ThreadPool()
{
    for (auto& t : m_threads)
        t.join();
}

void ThreadPool::stop()
{
    m_run.store(false);
}

void ThreadPool::submit(std::unique_ptr<RpcRequest> msg)
{
    std::lock_guard<std::mutex> lk(m_mtx);
    m_messages.push_back(std::move(msg));
    m_cond.notify_all();
}

void ThreadPool::run()
{
    while (m_run) {
        auto request = waitAndPop();
        if (!request)
            continue;
        request->callRpc();
    }
}

std::unique_ptr<RpcRequest> ThreadPool::waitAndPop()
{
    std::unique_lock<std::mutex> lk(m_mtx);
    m_cond.wait(lk, [this] { return !m_messages.empty(); });
    auto res = std::move(m_messages.front());
    m_messages.pop_front();
    return res;
}

std::unique_ptr<RpcRequest> ThreadPool::tryPop()
{
    std::lock_guard<std::mutex> lk(m_mtx);
    if (m_messages.empty())
        return std::unique_ptr<RpcRequest>();
    auto res = std::move(m_messages.front());
    m_messages.pop_front();
    return res;
}

bool ThreadPool::empty() const
{
    std::lock_guard<std::mutex> lk(m_mtx);
    return m_messages.empty();
}


}
}
