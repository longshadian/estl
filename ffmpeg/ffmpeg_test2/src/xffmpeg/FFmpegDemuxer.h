#pragma once

#include <vector>
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

class FFmpegDemuxer
{
public:
    typedef void (*DecodeProc)(AVCodecContext* ctx, AVFrame* frame, AVPacket* pkt, void* private_data);

public:
    FFmpegDemuxer();
    ~FFmpegDemuxer();

    int Init(AVCodecID codec_id = AV_CODEC_ID_H264);
    int ParsePkg(const void* buf, int len, int* consume_len);
    int ParsePkgAll(const void* buf, int len, DecodeProc proc, void* private_data);
    int Decode();

    /* const */ AVCodec* const_codec_;
    AVCodecParserContext* parser_;
    AVCodecContext* ctx_;
    AVPacket* pkt_;
    AVFrame* frame_;
};

