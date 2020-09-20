#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C"
{
#endif

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>

#ifdef __cplusplus
}
#endif

class MyFFmpegDemuxer
{
public:
    typedef void (*ParsePkgProc)(AVCodecContext* ctx, AVFrame* frame, AVPacket* pkt, void* private_data);
    typedef void (*DecodeFrameProc)(AVCodecContext* ctx, AVFrame* frame, void* private_data);

public:
    MyFFmpegDemuxer();
    ~MyFFmpegDemuxer();

    int Init(AVCodecID codec_id = AV_CODEC_ID_H264);
    int ParsePkg(const void* buf, int len, int* consume_len);
    int ParsePkgAll(const void* buf, int len, ParsePkgProc proc, void* private_data);
    int SendPacket();
    int Decode();
    int DecodeAll(DecodeFrameProc proc, void* private_data);
    int Demux(uint8_t** ppVideo, int* pnVideoBytes, int64_t* pts = NULL);

    /* const */ AVCodec* const_codec_;
    AVCodecParserContext* parser_;
    AVCodecContext* ctx_;
    AVPacket* pkt_;
    AVFrame* frame_;
    AVPacket* pktFiltered_;
    AVBSFContext* bsfc_;
    const uint8_t* data_;
    int data_size_;
    bool bMp4H264;
    bool bMp4HEVC;
    bool bMp4MPEG4;
};

