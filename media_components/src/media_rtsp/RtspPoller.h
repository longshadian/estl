#ifndef __RTSP_POLLER_H
#define __RTSP_POLLER_H

#include <memory>
#include <string>
#include <functional>

using GettingFrameProc 
    = std::function<void(
        unsigned char* /* buffer */,
        unsigned int /* buffer_length*/ ,
        unsigned /* numTruncatedBytes */,
        struct timeval /* presentationTime */,
        unsigned /* durationInMicroseconds */
         )>;

struct RtspPollerParams
{
    std::string url{};
    GettingFrameProc frame_proc{};
    std::unique_ptr<int> px;
};

class RtspPoller
{
    class RtspPollerImpl;
public:
    RtspPoller();
    ~RtspPoller();

    bool Init(RtspPollerParams argv);
    void Loop();

private:
    std::unique_ptr<RtspPollerImpl> impl_;
};

#endif