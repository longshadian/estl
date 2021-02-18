#ifndef _MEDIA_RTSP_RTSPSDK_H
#define _MEDIA_RTSP_RTSPSDK_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace media
{

// Video��������
enum ECodecType
{
    UnknownType = 0,
    H264 = 1000,
    HEVC = 2000,
};

// RTSP��������
enum EProtocolType
{
    UDP = 0,
    TCP = 1,
};

// δ����ԭʼ֡������Ϣ
struct RtspRawFrameInfo
{
    std::int32_t codec_type{};    // �����ʽ
    std::int32_t width{};         // ��
    std::int32_t height{};        // ��
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
