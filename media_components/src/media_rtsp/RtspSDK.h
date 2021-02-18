#ifndef _MEDIA_RTSP_RTSPSDK_H
#define _MEDIA_RTSP_RTSPSDK_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace media
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
struct RtspRawFrameInfo
{
    std::int32_t codec_type{};    // 编码格式
    std::int32_t width{};         // 宽
    std::int32_t height{};        // 高
};

using RtspHandle = std::uint32_t;

using FrameCallback
    = std::function<void(RtspHandle hdl, const RtspRawFrameInfo* info, std::uint8_t* buffer, std::int32_t buffer_length)>;

struct RtspParam
{
    std::int32_t protocol_type{};
};

class RtspSDK
{
public:
    class RtspSDKImpl;
    using Handle = void*;

    RtspSDK();
    ~RtspSDK();

    int Init();
    void Cleanup();
    int StartPullRtsp(const RtspParam* param, std::string url, FrameCallback cb, RtspHandle* hdl);
    int StopPullRtsp(RtspHandle hdl);

    Handle GetHandle() { return impl_.get(); }

private:
    std::unique_ptr<RtspSDKImpl> impl_;
};

} // namespace media

#endif
