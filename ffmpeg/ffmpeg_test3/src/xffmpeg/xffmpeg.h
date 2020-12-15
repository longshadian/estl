#pragma once

#include <cstdint>
#include <cstddef>
#include <sstream>
#include <string>

#ifdef __cplusplus
extern "C"
{
#endif

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>

#ifdef __cplusplus
}
#endif

namespace xffmpeg
{

int ffmpeg_save_jpeg(AVFrame* frame, const char* out_name);
void pgm_save(unsigned char* buf, int wrap, int xsize, int ysize, char* filename);

typedef void (*decode_proc)(AVCodecContext*, AVPacket*, AVFrame*, void*);
int decode(AVCodecContext* dec_ctx, AVPacket* pkt, AVFrame* frame, decode_proc proc = nullptr, void* pdata = nullptr);

void avpacket_to_string(const AVPacket* pkt, std::ostringstream& ostm);
void avctx_to_string(const AVCodecContext* ctx, std::ostringstream& ostm);
std::string picture_type(int v);

/**
 * 从内存中分流
 */
class MemoryDemuxer
{
public:
    MemoryDemuxer();
    ~MemoryDemuxer();

    int Init(AVCodecID codec_id = AV_CODEC_ID_H264);
    int Demux(uint8_t** ppVideo, int* pnVideoBytes);

    /* const */ AVCodec* const_codec_;
    AVCodecParserContext* parser_;
    AVCodecContext* ctx_;
    AVPacket* pkt_;
    AVFrame* frame_;
    AVPacket* pktFiltered_;
    AVBSFContext* bsfc_;
    const uint8_t* data_;
    int data_size_;
};

} // namespace xffmpeg


