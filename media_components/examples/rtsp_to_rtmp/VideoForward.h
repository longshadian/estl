#ifndef __VIDEO_FORWARD_H
#define __VIDEO_FORWARD_H

#include <memory>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <list>
#include <vector>
#include <cstddef>
#include <chrono>

#ifdef _WIN32
    #include <windows.h> //strcut timeval;
#endif // _WIN32

class SrsRtmp;
class RtspPoller;

struct VideoFrame
{
    uint64_t pts_;
    std::vector<char> buff;

    void Append(const char* p, size_t len, uint64_t pts)
    {
        buff.assign(p, p + len);
        pts_ = pts;
    }

    char* Data() { return buff.data(); }
    size_t Size() const { return buff.size(); }
    bool Emtpy() const { return buff.empty(); }
};
using VideoFramePtr = std::shared_ptr<VideoFrame>;

class VideoForward
{
public:
    struct RtspFrame 
    {
        unsigned char* buffer{};
        unsigned int buffer_length{};
        unsigned numTruncatedBytes{};
        struct timeval presentationTime{};
        unsigned durationInMicroseconds{};
        uint64_t pts{};
    };

public:
    VideoForward();
    virtual ~VideoForward();

    bool Init(const std::string& rtsp_uri, const std::string& rtmp_uri);
    void Loop();

protected:
    bool Init_rtsp();
    bool Init_rtmp();
    void FrameProc(unsigned char* buffer, unsigned int buffer_length, unsigned numTruncatedBytes, struct timeval presentationTime, unsigned durationInMicroseconds);

    void PostToRtmpQueue(const RtspFrame& info);
    void ProcessRtspFrame(VideoFramePtr& p);

    void RtmpThreadRun();

    VideoFramePtr PickupBuffer();
    void PutBuffer(VideoFramePtr b);

protected:
    std::string rtsp_uri_;
    std::string rtmp_uri_;

    std::unique_ptr<RtspPoller> rtsp_receiver_;
    std::unique_ptr<SrsRtmp> rtmp_sender_;

    bool rtmp_running_;
    std::thread rtmp_thread_;
    std::mutex mtx_;
    std::condition_variable cond_;
    std::list<VideoFramePtr> frame_buffer_;

    std::mutex cached_mtx_;
    std::list<VideoFramePtr> buffer_cached_; 

    uint64_t frame_num_;
    std::chrono::high_resolution_clock::time_point last_tp_;
    int32_t fps_;
    uint64_t pts_;
};

#endif // !__VIDEO_FORWARD_H
