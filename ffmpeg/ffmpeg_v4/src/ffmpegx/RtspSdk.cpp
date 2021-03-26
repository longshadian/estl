#include "ffmpegx/RtspSdk.h"

#include <string>
#include <iostream>
#include <sstream>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <cstring>

namespace ffmpegx
{

RtspSDK::RtspSDK()
    : running_{}
    , current_hdl_{}
    , mtx_{}
    , cond_{}
    , queue_{}
    , clients_{}
    , thd_{}
{
}

RtspSDK::~RtspSDK()
{
    Cleanup();
    if (thd_.joinable())
        thd_.join();
}

int RtspSDK::Init()
{
    running_ = true;
    std::thread temp_thd{
        [this] { this->ThreadRun(); }
    };
    std::swap(thd_, temp_thd);
    return 0;
}

void RtspSDK::Cleanup()
{
    running_ = false;
    PostTask([]{});
}

int RtspSDK::StartPullRtsp(const RtspParam& param, DataCallback sdk_cb, Handle* hdl)
{
    using namespace std::placeholders;
    auto new_hdl = ++current_hdl_;
    auto frame_cb = std::bind(&RtspSDK::OnRawFrameReceived, this, new_hdl, _1, _2, _3);
    std::unique_ptr<FFMpegClient> client{CreateClient(std::move(frame_cb), &param)};
    auto w = std::make_unique<ClientWrapper>(new_hdl, std::move(client), std::move(sdk_cb));

    auto ptask = std::make_unique<Task>();
    ptask->functor_ = std::bind(&RtspSDK::OnCreateClient, this, std::move(w));
    PostTask(std::move(ptask));
    *hdl = new_hdl;
    return 0;
}

int RtspSDK::StopPullRtsp(Handle hdl)
{
    PostTask([this, hdl] () mutable
    {
        this->ClientStopPull(hdl);
    });
    return 0;
}

void RtspSDK::ThreadRun()
{
    std::unique_ptr<Task> ptask{};
    while (running_) {
        {
            std::unique_lock<std::mutex> lk{mtx_};
            cond_.wait(lk, [this]() { return !queue_.empty(); });
            ptask = std::move(queue_.front());
            queue_.pop();
        }
        try {
            ptask->functor_();
        } catch (...) {
        }
    }
    clients_.clear();
}

void RtspSDK::ClientStopPull(Handle hdl)
{
    auto* client = FindClient(hdl);
    if (!client)
        return;
    client->Stop();
    clients_.erase(hdl);
}

void RtspSDK::OnCreateClient(std::unique_ptr<ClientWrapper> client)
{
    auto hdl = client->GetHandle();
    client->Init();
    clients_.emplace(hdl, std::move(client));
}

void RtspSDK::Async_RawFrameReceived(Handle hdl, const RawFrameInfo* info, const std::uint8_t* buffer, std::int32_t buffer_length)
{
    auto pinfo = std::make_unique<RawFrameInfo>(*info);
    std::vector<std::uint8_t> buf{buffer, buffer + buffer_length};
    auto ptask = std::make_unique<Task>();
    ptask->functor_ = std::bind(&RtspSDK::OnRawFrameReceived, std::move(pinfo), std::move(buf));
    PostTask(std::move(ptask));
}

void RtspSDK::OnRawFrameReceived(Handle hdl, std::unique_ptr<RawFrameInfo> info, std::vector<uint8_t> buffer)
{
    // 1. client不存在，移除此client
    // 2. client存在，执行user cb
    auto* client = FindClient(hdl); 
    if (!client) {
        client->Stop();
        clients_.erase(hdl);
        client = nullptr;
        return;
    }
    (*client)(hdl, info, buffer);
}

RtspSDK::ClientWrapper* RtspSDK::FindClient(Handle hdl)
{
    auto it = clients_.find(hdl);
    if (it == clients_.end())
        return nullptr;
    return it->second.get();
}

void RtspSDK::PostTask(std::function<void> f)
{
    auto t = std::make_unique<Task>();
    t->functor_ = std::move(f);
    PostTask(std::move(t));
}

void RtspSDK::PostTask(std::unique_ptr<Task> t)
{
    std::lock_guard<std::mutex> lk{mtx_};
    queue_.emplace(std::move(t));
    cond_.notify_one();
}

} // namespace ffmpegx

