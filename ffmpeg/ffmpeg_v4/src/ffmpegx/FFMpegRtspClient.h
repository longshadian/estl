#ifndef _FFMPEGX_CLIENT_H
#define _FFMPGEX_CLIENT_H

#include <atomic>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>
#include <unordered_map>
#include <queue>

#ifdef __cplusplus
extern "C"
{
#endif

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>

#ifdef __cplusplus
}
#endif


namespace ffmpegx
{

// Video解码类型
enum ECodecType
{
    UnknownType = 0,
    H264 = 1000,
    HEVC = 2000,
};

// RTSP传输类型
enum EProtocolType
{
    UDP = 0,
    TCP = 1,
};

// 未解码原始帧数据信息
struct RawFrameInfo
{
    std::int32_t codec_type{};    // 编码格式
    std::int32_t width{};         // 宽
    std::int32_t height{};        // 高
};

using RtspHandle = std::uint32_t;

using FrameCallback
    = std::function<void(const RawFrameInfo* info, std::uint8_t* buffer, std::int32_t buffer_length)>;

struct RtspParam
{
    std::string url{};
    EProtocolType protocol_type{};
    std::chrono::milliseconds connect_timeout{ std::chrono::seconds{5} };
    std::chrono::milliseconds read_timeout{ std::chrono::seconds{5} };
};

struct RtmpParam
{
    std::string url{};
    std::chrono::milliseconds connect_timeout{ std::chrono::seconds{5} };
    std::chrono::milliseconds read_timeout{ std::chrono::seconds{5} };
};

class FFMpegClient
{
    using Clock = std::chrono::steady_clock;
public:
    FFMpegClient(FrameCallback cb);
    virtual ~FFMpegClient();

    virtual int Init() = 0;
    void Stop() { running_ = 0; }
    bool Stopped() const { return running_; }
    virtual int Loop();
    virtual int OnReadPkt(AVPacket* pkt);

    static int CheckInterrupt(void* ctx);
    int CheckInterruptEx();

public:
    AVFormatContext* ifmt_ctx_;
    AVDictionary*   options_;
    AVPacket* pkt_;
    bool running_;
    bool inited_;
    Clock::time_point last_read_tp_;
    FrameCallback  cb_;

protected:
    std::chrono::milliseconds read_timeout_;
    std::string url_;
};

FFMpegClient* CreateClient(FrameCallback cb, const RtspParam*  param);
FFMpegClient* CreateClient(FrameCallback cb, const RtmpParam*  param);

/*
class FFMpegSdk
{
public:
    FFMpegSdk();
    ~FFMpegSdk();

    int Init();
    void Cleanup();
    int StartPullRtsp(const RtspParam* param, FrameCallback cb, RtspHandle* hdl);
    int StopPullRtsp(RtspHandle hdl);

private:
    std::mutex mtx_;

};
*/

} // namespace ffmpegx

#endif // !_FFMPEGX_CLIENT_H

