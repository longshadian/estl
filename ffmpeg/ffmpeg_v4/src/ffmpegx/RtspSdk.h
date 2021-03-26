#ifndef _FFMPEGX_RTSPSDK_H
#define _FFMPEGX_RTSPSDK_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "ffmpegx/RtspClient.h"

namespace ffmpegx
{

class RtspSDK
{
public:
    struct Task
    {
        std::function<void()> functor_;
    };

    using Handle = std::uint32_t;
    using DataCallback = std::function<void(Handle hdl, std::unique_ptr<RawFrameInfo>& info, std::vector<std::uint8_t>& buffer)>;

private:
    class ClientWrapper
    {
    public:
        ClientWrapper(Handle hdl, std::unique_ptr<FFMpegClient> client, DataCallback frame_cb)
            : hdl_{hdl}
            , client_{std::move(client)}
            , thd_{}
            , data_cb_{std::move(frame_cb)}
        {
        }

        ~ClientWrapper()
        {
            Stop();
            if (thd_.joinable())
                thd_.join();
        }

        ClientWrapper(const ClientWrapper&) = delete;
        ClientWrapper& operator=(const ClientWrapper&) = delete;

        int Init()
        {
            try {
                std::thread tmp_thd{[this]() { client_->Loop(); }};
                std::swap(thd_, tmp_thd);
            } catch (...) {
                return -1;
            }
            return 0;
        }

        void Stop()
        {
            client_->Stop();
        }

        //void OnReceived(std::unique_ptr<RawFrameInfo> info, std::vector<> )

        FFMpegClient* GetImpl() { return client_.get(); }

        void operator()(Handle hdl, std::unique_ptr<RawFrameInfo>& info, std::vector<std::uint8_t>& buffer)
        {
            try {
                data_cb_(hdl, info, buffer);
            } catch(...) {
            }
        }

        Handle GetHandle() const { return hdl_; }

    private:
        Handle hdl_;
        std::unique_ptr<FFMpegClient> client_;
        std::thread thd_;
        DataCallback data_cb_;
    };

public:
    RtspSDK();
    ~RtspSDK();

    int Init();
    void Cleanup();
    int StartPullRtsp(const RtspParam& param, DataCallback user_cb, Handle* hdl);
    int StopPullRtsp(Handle hdl);

private:
    void ThreadRun();
    void ClientStopPull(Handle hdl);
    void OnCreateClient(std::unique_ptr<ClientWrapper> client);
    void Async_RawFrameReceived(Handle hdl, const RawFrameInfo* info, const std::uint8_t* buffer, std::int32_t buffer_length);
    void OnRawFrameReceived(Handle hdl, const RawFrameInfo* info, const std::uint8_t* buffer, std::int32_t buffer_length);

    ClientWrapper* FindClient(Handle hdl);
    void PostTask(std::function<void()> f);
    void PostTask(std::unique_ptr<Task> t);

private:
    std::atomic<bool>                   running_;
    std::atomic<Handle>                 current_hdl_;
    std::mutex                          mtx_;
    std::condition_variable             cond_;
    std::queue<std::unique_ptr<Task>>   queue_;
    std::unordered_map<Handle, std::unique_ptr<ClientWrapper>> clients_;
    std::thread                         thd_;
};

} // namespace ffmpegx

#endif
