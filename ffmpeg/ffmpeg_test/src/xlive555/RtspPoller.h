#ifndef __RTSP_POLLER_H
#define __RTSP_POLLER_H

#include <memory>
#include <string>

typedef void (*GettingFrameProc)(
    unsigned char* buffer,
    unsigned int buffer_length,
    unsigned numTruncatedBytes,
    struct timeval presentationTime,
    unsigned durationInMicroseconds,
    void* private_data
);

struct RtspPollerParams
{
    std::string url{};
    GettingFrameProc frame_proc{};
    void* frame_proc_private_data{};       
};

class RtspPollerImpl;
class RtspPoller
{
public:
    RtspPoller();
    ~RtspPoller();

    bool Init(const RtspPollerParams& argv);
    void Loop();

private:
    std::unique_ptr<RtspPollerImpl> impl_;
};

#endif