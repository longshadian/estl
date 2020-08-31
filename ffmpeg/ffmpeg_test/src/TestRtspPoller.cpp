#include <iostream>
#include <sstream>

#include <windows.h>

#include "xlive555/RtspPoller.h"

static
void FrameProc(
    unsigned char* buffer,
    unsigned int buffer_length,
    unsigned numTruncatedBytes,
    struct timeval presentationTime,
    unsigned durationInMicroseconds,
    void* private_data
    )
{
    static int n = 0;
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
}

int TestRtspPoller()
{
    std::string url = "rtmp://192.168.1.95:31935/videotest/test";

    RtspPoller impl;
    RtspPollerParams params;
    params.url = "rtsp://192.168.1.95:8554/a.264";
    params.frame_proc = FrameProc;
    params.frame_proc_private_data = &impl;
    if (!impl.Init(params)) {
        std::cout << "init error\n";
        return 0;
    }
    impl.Loop();

    return 0;
}




