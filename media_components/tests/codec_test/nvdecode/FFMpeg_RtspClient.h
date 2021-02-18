#pragma once

#include <atomic>
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

#include "console_log.h"
#include "RtspClient.h"

class FFMpeg_Base
{
public:
    FFMpeg_Base();
    virtual ~FFMpeg_Base();

    virtual int Init(std::string url, Callback_t cb);
    virtual int Loop();

    Callback_t cb_;
    std::string url_;
};

class FFMpeg_File : public FFMpeg_Base
{
    using Super = FFMpeg_Base;
public:
    FFMpeg_File();
    ~FFMpeg_File();

    virtual int Init(std::string url, Callback_t cb) override;
    virtual int Loop() override;

    int ParserDecode(const void* p, size_t size);

    xffmpeg::MemParser demuxer_;
};

class FFMpeg_Rtsp : public FFMpeg_File
{
    using Super = FFMpeg_File;
public:
    FFMpeg_Rtsp();
    ~FFMpeg_Rtsp();

    virtual int Init(std::string url, Callback_t cb) override;
    virtual int Loop() override;
    virtual int OnReadPkt(AVPacket* pkt) =0;

    AVFormatContext* ifmt_ctx_;
    int             stream_index_;
    const AVStream*  vid_stream_;
};

class FFMpeg_Rtsp_Decode : public FFMpeg_Rtsp
{
    using Super = FFMpeg_Rtsp;
public:
    FFMpeg_Rtsp_Decode();
    ~FFMpeg_Rtsp_Decode();

    virtual int Init(std::string url, Callback_t cb) override;
    virtual int OnReadPkt(AVPacket* pkt) override;
    void decode_proc(AVCodecContext*, AVPacket*, AVFrame* frame);

    const AVCodec*          const_codec_{};
    AVCodecContext*         dec_ctx_{};
    AVFrame*                dec_frame_{};
};

class FFMpeg_Rtsp_Decode2 : public FFMpeg_Rtsp_Decode
{
    using Super = FFMpeg_Rtsp_Decode;
public:
    FFMpeg_Rtsp_Decode2();
    ~FFMpeg_Rtsp_Decode2();

    virtual int Loop() override;
    void decode_proc(AVCodecContext*, AVPacket*, AVFrame* frame);
};
