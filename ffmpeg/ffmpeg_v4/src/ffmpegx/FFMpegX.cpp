#include "ffmpegx/FFMpegX.h"

#include <string>
#include <sstream>

namespace ffmpegx
{

int ffmpeg_save_jpeg(AVFrame* frame, const char* out_name)
{
    int ret = 0;
    int width = frame->width;
    int height = frame->height;
    const AVCodec* codec = NULL;
    AVFormatContext* fmt_ctx = NULL;
    AVCodecContext* codec_ctx = NULL;
    AVStream* stream = NULL;
    AVCodecParameters* parameters = NULL;
    AVPacket pkt;

    do {
        // 设置输出文件格式
        fmt_ctx = avformat_alloc_context();
        fmt_ctx->oformat = av_guess_format("mjpeg", NULL, NULL);

        // 创建并初始化输出AVIOContext
        ret = avio_open(&fmt_ctx->pb, out_name, AVIO_FLAG_READ_WRITE);
        if (ret < 0) {
            break;
        }

        // 构建一个新stream
        stream = avformat_new_stream(fmt_ctx, 0);
        if (!stream) {
            ret = -1;
            break;
        }

        parameters = stream->codecpar;
        parameters->codec_id = fmt_ctx->oformat->video_codec;
        parameters->codec_type = AVMEDIA_TYPE_VIDEO;
        parameters->format = AV_PIX_FMT_YUVJ420P;
        parameters->width = frame->width;
        parameters->height = frame->height;

        codec = avcodec_find_encoder(stream->codecpar->codec_id);
        if (!codec) {
            ret = -1;
            break;
        }
        codec_ctx = avcodec_alloc_context3(codec);
        if (!codec_ctx) {
            ret = -1;
            break;
        }

        ret = avcodec_parameters_to_context(codec_ctx, stream->codecpar);
        if (ret < 0) {
            //fprintf(stderr, "Failed to copy %s codec parameters to decoder context\n", av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
            break;
        }

        //codec_ctx->time_base = (AVRational){ 1, 25 };
        codec_ctx->time_base.num = 1;
        codec_ctx->time_base.den = 25;

        ret = avcodec_open2(codec_ctx, codec, NULL);
        if (ret < 0) {
            //printf("Could not open codec.");
            break;
        }

        ret = avformat_write_header(fmt_ctx, NULL);
        if (ret < 0) {
            //printf("write_header fail\n");
            break;
        }

        // Encode
        // 给AVPacket分配足够大的空间
        ret = av_new_packet(&pkt, width * height * 3);
        if (ret != 0) {
            break;
        }

        // 编码数据
        ret = avcodec_send_frame(codec_ctx, frame);
        if (ret < 0) {
            //printf("Could not avcodec_send_frame.");
            break;
        }

        // 得到编码后数据
        ret = avcodec_receive_packet(codec_ctx, &pkt);
        if (ret < 0) {
            //printf("Could not avcodec_receive_packet");
            break;
        }

        ret = av_write_frame(fmt_ctx, &pkt);
        if (ret < 0) {
            //printf("Could not av_write_frame");
            break;
        }

        //Write Trailer
        ret = av_write_trailer(fmt_ctx);
        if (ret < 0) {
            break;
        }
    } while (0);

    av_packet_unref(&pkt);
    avio_close(fmt_ctx->pb);
    avcodec_free_context(&codec_ctx);
    avformat_free_context(fmt_ctx);
    return ret;
}

void pgm_save(unsigned char* buf, int wrap, int xsize, int ysize, char* filename)
{
    FILE* f;
    int i;
    f = fopen(filename, "w");
    fprintf(f, "P5\n%d %d\n%d\n", xsize, ysize, 255);
    for (i = 0; i < ysize; i++)
        fwrite(buf + i * wrap, 1, xsize, f);
    fclose(f);
}

int decode(AVCodecContext* dec_ctx, AVPacket* pkt, AVFrame* frame, decode_proc proc, void* pdata)
{
    int ret = avcodec_send_packet(dec_ctx, pkt);
    if (ret < 0) {
        printf("Error sending a packet for decoding\n");
        return -1;
    }
    while (ret >= 0) {
        ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return 0;
        else if (ret < 0) {
            printf("Error during decoding\n");
            return -1;
        }
        if (proc)
            proc(dec_ctx, pkt, frame, pdata);
    }
    return 0;
}


void AVPacket_to_string(const AVPacket* pkt, std::ostringstream& ostm)
{
    std::string tab = "     ";
    std::string crlf = " ";
    ostm << "AVPacket:\n"
        << tab << "pts:         " << pkt->pts << crlf
        << tab << "dts:         " << pkt->dts << crlf
        << tab << "size:        " << pkt->size << crlf
        << tab << "stream_idx:  " << pkt->stream_index << crlf
        << tab << "flags:       " << pkt->flags << crlf
        << tab << "side_de:     " << pkt->side_data_elems << crlf
        << tab << "duration:    " << pkt->duration << crlf
        << tab << "pos:         " << pkt->pos << crlf
        ;
}

void AVCodecContext_to_string(const AVCodecContext* ctx, std::ostringstream& ostm)
{
    std::string tab = "     ";
    std::string crlf = " ";
    ostm << "AVCodecContext:" << crlf
        << tab << "codec_tag:         " << ctx->codec_tag << crlf
        << tab << "bit_rate:         " << ctx->bit_rate << crlf
        << tab << "bit_rate_tolerance:        " << ctx->bit_rate_tolerance << crlf
        << tab << "global_quality:  " << ctx->global_quality << crlf
        << tab << "compression_level:       " << ctx->compression_level << crlf
        << tab << "ticks_per_frame:     " << ctx->ticks_per_frame << crlf
        << tab << "width:    " << ctx->width << crlf
        << tab << "height:         " << ctx->height << crlf
        << tab << "coded_width:    " << ctx->coded_width << crlf
        << tab << "coded_height:    " << ctx->coded_height << crlf

        << tab << "gop_size:    " << ctx->gop_size << crlf
        << tab << "pix_fmt:    " << ctx->pix_fmt << crlf
        ;
}

void AVCodecParameters_to_string(const AVCodecParameters* p, std::ostringstream& ostm)
{
    std::string tab = "     ";
    std::string crlf = "\n";
    ostm << "AVCodecParameters:" << crlf
        << tab << "codec_type:          " << p->codec_type << crlf
        << tab << "codec_id:            " << p->codec_id << crlf
        << tab << "codec_tag:           " << p->codec_tag << crlf
        << tab << "format:              " << p->format << crlf
        << tab << "bit_rate:            " << p->bit_rate << crlf
        << tab << "bits_per_coded_sample:   " << p->bits_per_coded_sample << crlf
        << tab << "bits_per_raw_sample:     " << p->bits_per_raw_sample << crlf
        << tab << "profile:                 " << p->profile << crlf
        << tab << "level:               " << p->level << crlf
        << tab << "width:               " << p->width << crlf
        << tab << "height:              " << p->height << crlf
        << tab << "sample_aspect_ratio: " << p->sample_aspect_ratio.den << " " << p->sample_aspect_ratio.num << crlf
        << tab << "field_order:         " << p->field_order << crlf

        << tab << "color_range:         " << p->color_range << crlf
        << tab << "color_primaries:     " << p->color_primaries << crlf
        << tab << "color_trc:           " << p->color_trc << crlf
        << tab << "color_space:         " << p->color_space << crlf
        << tab << "chroma_location:     " << p->chroma_location << crlf
        << tab << "video_delay:         " << p->video_delay << crlf
        << tab << "channel_layout:      " << p->channel_layout << crlf
        << tab << "channels:            " << p->channels << crlf
        << tab << "sample_rate:         " << p->sample_rate << crlf
        << tab << "block_align:         " << p->block_align << crlf
        << tab << "frame_size:          " << p->frame_size << crlf
        << tab << "initial_padding:     " << p->initial_padding << crlf
        << tab << "trailing_padding:    " << p->trailing_padding << crlf
        << tab << "seek_preroll:        " << p->seek_preroll << crlf
        ;
}

std::string picture_type(int v)
{
    switch (v)
    {
    case AV_PICTURE_TYPE_NONE: return "Undefined";
    case AV_PICTURE_TYPE_I: return "Intra";
    case AV_PICTURE_TYPE_P: return "Predicted";
    case AV_PICTURE_TYPE_B: return "Bi-dir predicted";
    case AV_PICTURE_TYPE_S: return "S(GMC)-VOP MPEG-4";
    case AV_PICTURE_TYPE_SI: return "Switching Intra";
    case AV_PICTURE_TYPE_SP: return "Switching Predicted";
    case AV_PICTURE_TYPE_BI: return "BI type";
    default:
        break;
    }
    return "??";
}


//=======================================================================
// class MemParser
//=======================================================================

MemParser::MemParser()
    : const_codec_{}
    , codec_parser_ctx_{}
    , codec_context_{}
    , packet_{}
    , frame_{}
    , codec_parameters_{}
    , raw_data_{}
    , raw_data_size_{}
{
}

MemParser::~MemParser()
{
    if (codec_parser_ctx_)
        ::av_parser_close(codec_parser_ctx_);
    if (codec_context_)
        ::avcodec_free_context(&codec_context_);
    if (packet_)
        ::av_packet_free(&packet_);
    if (frame_)
        ::av_frame_free(&frame_);
    if (codec_parameters_)
        ::avcodec_parameters_free(&codec_parameters_);
}

int MemParser::Init_H264()
{
    return Init(AV_CODEC_ID_H264);
}

int MemParser::Init_H265()
{
    return Init(AV_CODEC_ID_H265);
}

int MemParser::Init(AVCodecID codec_id)
{
    const_codec_ = ::avcodec_find_decoder(codec_id);
    if (!const_codec_) {
        return -1;
    }
    codec_parser_ctx_ = ::av_parser_init(const_codec_->id);
    if (!codec_parser_ctx_) {
        return -1;
    }
    codec_context_ = ::avcodec_alloc_context3(const_codec_);
    if (!codec_context_) {
        return -1;
    }

    /* For some codecs, such as msmpeg4 and mpeg4, width and height
       MUST be initialized there because this information is not
       available in the bitstream. */
       /* open it */
    if (::avcodec_open2(codec_context_, const_codec_, NULL) < 0) {
        return -1;
    }
    packet_ = ::av_packet_alloc();
    if (!packet_)
        return -1;
    frame_ = ::av_frame_alloc();
    if (!frame_)
        return -1;
    codec_parameters_ = ::avcodec_parameters_alloc();
    if (!codec_parameters_)
        return -1;
    return 0;
}

void MemParser::AppendRawData(const void* data, size_t size)
{
    raw_data_ = static_cast<const uint8_t*>(data);
    raw_data_size_ = static_cast<int>(size);
}

int MemParser::Parse()
{
    bool has_pkg = false;
    int len = 0;
    while (raw_data_size_ > 0) {
        len = ::av_parser_parse2(codec_parser_ctx_, codec_context_, &packet_->data, &packet_->size,
            raw_data_, raw_data_size_, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
        if (len < 0) {
            return -1;
        }
        raw_data_ += len;
        raw_data_size_ -= len;
        if (packet_->size) {
            has_pkg = true;
            break;
        }
    }
    if (!has_pkg)
        return 0;
    return 1;
}

int MemParser::PrepareDecode()
{
    int ecode = ::avcodec_send_packet(codec_context_, packet_);
    if (ecode == AVERROR(EAGAIN) || ecode == AVERROR_EOF)
        return 0;
    return ecode;
}

int MemParser::PrepareCodecParameter()
{
    return ::avcodec_parameters_from_context(codec_parameters_, codec_context_);
}

int MemParser::Decode()
{
    int ecode = ::avcodec_receive_frame(codec_context_, frame_);
    if (ecode == AVERROR(EAGAIN) || ecode == AVERROR_EOF)
        return 0;
    else if (ecode < 0)
        return -1;
    else
        return 1;
}

} // namespace ffmpegx


