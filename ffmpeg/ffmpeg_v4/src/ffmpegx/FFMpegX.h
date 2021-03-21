#ifndef _FFMPGE_TOOLS_H
#define _FFMPGE_TOOLS_H

#include <cstdint>
#include <cstddef>
#include <sstream>
#include <string>
#include <chrono>

#ifdef __cplusplus
extern "C"
{
#endif

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>

#ifdef __cplusplus
}
#endif

namespace ffmpegx
{

int ffmpeg_save_jpeg(AVFrame* frame, const char* out_name);
void pgm_save(unsigned char* buf, int wrap, int xsize, int ysize, char* filename);

typedef void (*decode_proc)(AVCodecContext*, AVPacket*, AVFrame*, void*);
int decode(AVCodecContext* dec_ctx, AVPacket* pkt, AVFrame* frame, decode_proc proc = nullptr, void* pdata = nullptr);

void AVPacket_to_string(const AVPacket* pkt, std::ostringstream& ostm);
void AVCodecContext_to_string(const AVCodecContext* ctx, std::ostringstream& ostm);
void AVCodecParameters_to_string(const AVCodecParameters* p, std::ostringstream& ostm);
std::string picture_type(int v);

std::string picture_type(int v);

struct MicrosecondTimer
{
    MicrosecondTimer() : tb_(), te_() {}

    void Start() { tb_ = std::chrono::steady_clock::now(); }
    void Stop() { te_ = std::chrono::steady_clock::now(); }
    std::int64_t Delta() const { return std::chrono::duration_cast<std::chrono::microseconds>(te_ - tb_).count(); }
    float GetMilliseconds() const { return static_cast<float>(Delta()) / 1000.f; }

    std::chrono::steady_clock::time_point tb_;
    std::chrono::steady_clock::time_point te_;
};

class MemParser
{
public:
    MemParser();
    ~MemParser();

    int Init(AVCodecID codec_id);
    int Init_H264();
    int Init_H265();

    void AppendRawData(const void* data, size_t size);
    int Parse();
    int PrepareDecode();
    int PrepareCodecParameter();
    int Decode();

    AVCodecContext* GetCodecContext() { return codec_context_; }
    AVPacket* GetPacket() { return packet_; }
    AVFrame* GetFrame() { return frame_; }
    AVCodecParameters* GetCodecParameters() { return codec_parameters_; }

    const AVCodec*          const_codec_;
    AVCodecParserContext*   codec_parser_ctx_;
    AVCodecContext*         codec_context_;
    AVPacket*               packet_;
    AVFrame*                frame_;
    AVCodecParameters*      codec_parameters_;
    const uint8_t*          raw_data_;
    int                     raw_data_size_;
};

} // namespace ffmpegx

#endif // !_FFMPGE_TOOLS_H

