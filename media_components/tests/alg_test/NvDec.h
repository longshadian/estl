#pragma once

#include <condition_variable>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include <RtspPoller.h>

#include <NvCodec/NvDecoder/NvDecoder.h>

#include "algMultiServer.h"

class NvDec
{
public:
    struct FrameData
    {
        int64_t frame_num_{};
        std::vector<uint8_t> data_{};
        unsigned numTruncatedBytes_{};
        struct timeval presentationTime_{};
        unsigned durationInMicroseconds_{};
    };

public:
    NvDec();
    ~NvDec();

    void FrameProc(
        unsigned char* buffer,
        unsigned int buffer_length,
        unsigned numTruncatedBytes,
        struct timeval presentationTime,
        unsigned durationInMicroseconds
    );

    int Init(int gpu_num, cudaVideoCodec type);
    int StartPullRtspThread(std::string rtsp_uri);
    int StartDecoder(std::string pic_dir);

private:
    int VideoDecode(FrameData& frame_data, const std::string& pic_dir);

    int frame_num_;
    CUcontext cuContext;
    std::unique_ptr<RtspPoller> rtsp_;
    std::unique_ptr<NvDecoder> nvdec_;
    std::thread rtsp_thd_;

    std::mutex mtx_;
    std::list<std::unique_ptr<FrameData>> queue_;
    std::condition_variable cond_;
    Rect cropRect_;
    Dim resizeDim_;

    std::unique_ptr<algMultiServer> server_;

    uint64_t gpu_buffer_;
}; 


