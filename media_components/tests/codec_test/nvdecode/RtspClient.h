#pragma once

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include <media_rtsp/RtspSDK.h>
#include <xffmpeg/xffmpeg.h>

using Callback_t = std::function<void(
    unsigned char* buffer,
    unsigned int buffer_length,
    unsigned numTruncatedBytes,
    struct timeval presentationTime,
    unsigned durationInMicroseconds
)>;

class RtspClient_Live555
{
public:
    RtspClient_Live555();
    ~RtspClient_Live555();

    int Init(std::string url, Callback_t cb);
    int Loop();

    void FrameProcess(
        unsigned char* buffer,
        unsigned int buffer_length,
        unsigned numTruncatedBytes,
        struct timeval presentationTime,
        unsigned durationInMicroseconds
    );

    void FrameCB(media::RtspHandle hdl, const media::RtspRawFrameInfo*, std::uint8_t* buffer, std::int32_t buffer_length);

    std::unique_ptr<media::RtspSDK> sdk_;
    Callback_t cb_;
    media::RtspHandle hdl_;
    xffmpeg::MemParser parser_;
    int num_;
};

class RtspClient_File_HEVC
{
public:
    RtspClient_File_HEVC();
    ~RtspClient_File_HEVC();

    int Init(std::string url, Callback_t cb);
    int Loop();

    Callback_t cb_;
    std::string fname_;
};


class RtspClient_HEVC_Save
{
public:
    RtspClient_HEVC_Save();
    ~RtspClient_HEVC_Save();

    int Init(std::string url, Callback_t cb);
    int Loop();
    int Fun();

    void FrameProcess(
        unsigned char* buffer,
        unsigned int buffer_length,
        unsigned numTruncatedBytes,
        struct timeval presentationTime,
        unsigned durationInMicroseconds
    );

    Callback_t cb_;
    std::string fname_;
    AVFormatContext* ifmt_ctx;
    int* stream_mapping;
    int stream_index;
    int stream_mapping_size;
    double duration;

    AVFormatContext* ofmt_ctx;
    AVStream* out_stream;
};

