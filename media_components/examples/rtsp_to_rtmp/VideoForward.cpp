#include "VideoForward.h"

#include <chrono>
#include <cassert>
#include <thread>

#include <RtspPoller.h>
#include <SRSRtmp.h>
#include "console_log.h"

VideoForward::VideoForward()
    : rtsp_uri_()
    , rtmp_uri_()
    , rtsp_receiver_(std::make_unique<RtspPoller>())
    , rtmp_sender_(std::make_unique<SrsRtmp>())
    , rtmp_running_()
    , rtmp_thread_()
    , mtx_()
    , cond_()
    , frame_buffer_()
    , cached_mtx_()
    , buffer_cached_()
    , frame_num_()
    , last_tp_(std::chrono::high_resolution_clock::now())
    , fps_(30)
    , pts_()
{
}

VideoForward::~VideoForward()
{
    rtmp_running_ = false;
    if (rtmp_thread_.joinable())
        rtmp_thread_.join();
}

bool VideoForward::Init(const std::string& rtsp_uri, const std::string& rtmp_uri)
{
    rtsp_uri_ = rtsp_uri;
    rtmp_uri_ = rtmp_uri;
    if (!Init_rtmp())
        return false;
    if (!Init_rtsp())
        return false;
    return true;
}

void VideoForward::Loop()
{
#if 0
    int tick = 0;
    while (1) {
        tick++;
        std::this_thread::sleep_for(std::chrono::seconds(5));
        logPrintInfo("tick: %d", tick);
    }
#endif
    rtsp_receiver_->Loop();
}

bool VideoForward::Init_rtsp()
{
    using namespace std::placeholders;

    RtspPollerParams params;
    params.url = rtsp_uri_;
    params.frame_proc = std::bind(&VideoForward::FrameProc, this, _1, _2, _3, _4, _5);
    if (!rtsp_receiver_->Init(std::move(params))) {
        std::cout << "init error\n";
        return false;
    }
    return true;
}

bool VideoForward::Init_rtmp()
{
    rtmp_sender_->Reset(rtmp_uri_);
    if (rtmp_sender_->Init() != 0) {
        return false;
    }

    rtmp_running_ = true;
    std::thread tmp(&VideoForward::RtmpThreadRun, this);
    rtmp_thread_ = std::move(tmp);
    return true;
}

void VideoForward::FrameProc(
    unsigned char* buffer,
    unsigned int buffer_length,
    unsigned numTruncatedBytes,
    struct timeval presentationTime,
    unsigned durationInMicroseconds
)
{
    if (0) {
        int a0 = buffer[0];
        int a1 = buffer[1];
        int a2 = buffer[2];
        int a3 = buffer[3];
        printf("FrameProc: %x %x %x %x\n", a0, a1, a2, a3);
    }

#if 0
    std::ostringstream ostm{};
    ostm << "FrameProc: " << ++n
        << " buffer_length: " << buffer_length
        << "\n";
    if (numTruncatedBytes > 0)
        ostm << " (with " << numTruncatedBytes << " bytes truncated)";
    char uSecsStr[6 + 1]; // used to output the 'microseconds' part of the presentation time
    sprintf(uSecsStr, "%06u", (unsigned)presentationTime.tv_usec);
    ostm << ".\tPresentation time: " << (int)presentationTime.tv_sec << "." << uSecsStr;
    ostm << "\n";
    std::cout << ostm.str();
#endif
    if (frame_num_ % 100 == 0) {
        if (frame_num_ == 0) {
            last_tp_ = std::chrono::high_resolution_clock::now();
        } else {
            auto tnow = std::chrono::high_resolution_clock::now();
            int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(tnow - last_tp_).count();
            last_tp_ = tnow;
            fps_ = ms / 100;
            if (fps_ <= 0)
                fps_ = 30;
        }
    }
    ++frame_num_;
    pts_ += fps_;
    logPrintInfo("frame: %d fps: %d pts: %d", (int)frame_num_, (int)fps_, (int)pts_);

    RtspFrame frame;
    frame.buffer = buffer;
    frame.buffer_length = buffer_length;
    frame.durationInMicroseconds = durationInMicroseconds;
    frame.numTruncatedBytes = numTruncatedBytes;
    frame.presentationTime = presentationTime;
    frame.pts = pts_;
    PostToRtmpQueue(frame);
}

void VideoForward::PostToRtmpQueue(const RtspFrame& info)
{
    auto frame = PickupBuffer();
    frame->Append((const char*)info.buffer, info.buffer_length, info.pts);
    {
        std::lock_guard<std::mutex> lk{mtx_};
        frame_buffer_.push_back(frame);
    }
}

void VideoForward::ProcessRtspFrame(VideoFramePtr& p)
{
    int ret = rtmp_sender_->SendH264(p->Data(), p->Size(), p->pts_);
    if (ret == -1) {
        rtmp_sender_->Reset();
        ret = rtmp_sender_->Init();
    }

    if (ret != 0)
        logPrintWarn("ProcessRtspFrame error %d\n", ret);
}

void VideoForward::RtmpThreadRun()
{
    VideoFramePtr frame;
    while (rtmp_running_) {
        {
            std::unique_lock<std::mutex> lk{mtx_};
            cond_.wait_for(lk, std::chrono::seconds{1}, [this]{ return !frame_buffer_.empty(); });
            if (frame_buffer_.empty())
                continue;
            frame = frame_buffer_.front();
            frame_buffer_.pop_front();
        }

        assert(frame);
        ProcessRtspFrame(frame);
        PutBuffer(frame);
        frame = nullptr;
    }
}

VideoFramePtr VideoForward::PickupBuffer()
{
    {
        std::lock_guard<std::mutex> lk{cached_mtx_};
        if (!buffer_cached_.empty()) {
            VideoFramePtr p = buffer_cached_.front();
            buffer_cached_.pop_front();
            return p;
        }
    }
    return std::make_shared<VideoFrame>();
}

void VideoForward::PutBuffer(VideoFramePtr b)
{
    std::lock_guard<std::mutex> lk{cached_mtx_};
    buffer_cached_.push_back(b);
}

