#include "xffmpeg/FFmpegDemuxer.h"


FFmpegDemuxer::FFmpegDemuxer()
    : const_codec_()
    , parser_()
    , ctx_()
    , pkt_()
    , frame_()
{
}

FFmpegDemuxer::~FFmpegDemuxer()
{
    if (parser_)
        ::av_parser_close(parser_);
    if (ctx_)
        ::avcodec_free_context(&ctx_);
    if (pkt_)
        ::av_packet_free(&pkt_);
    if (frame_)
        ::av_frame_free(&frame_);
}

int FFmpegDemuxer::Init(AVCodecID codec_id)
{
    const_codec_ = ::avcodec_find_decoder(codec_id);
    if (!const_codec_) {
        return -1;
    }
    parser_ = ::av_parser_init(const_codec_->id);
    if (!parser_) {
        return -1;
    }
    ctx_ = ::avcodec_alloc_context3(const_codec_);
    if (!ctx_) {
        return -1;
    }

    /* For some codecs, such as msmpeg4 and mpeg4, width and height
       MUST be initialized there because this information is not
       available in the bitstream. */
       /* open it */
    if (::avcodec_open2(ctx_, const_codec_, NULL) < 0) {
        return -1;
    }
    pkt_ = ::av_packet_alloc();
    if (!pkt_)
        return -1;
    frame_ = ::av_frame_alloc();
    if (!frame_) {
        return -1;
    }
    return 0;
}

int FFmpegDemuxer::ParsePkg(const void* buf, int len, int* consume_len) 
{
    const uint8_t* data = reinterpret_cast<const uint8_t*>(buf);
    int ret = ::av_parser_parse2(parser_, ctx_, &pkt_->data, &pkt_->size,
            data, len, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
    if (ret < 0) {
        return -1;
    }
    *consume_len = ret;
    return pkt_->size == 0  ? 0 : 1;
}

int FFmpegDemuxer::ParsePkgAll(const void* buf, int len, ParsePkgProc proc, void* private_data)
{
    const uint8_t* data = reinterpret_cast<const uint8_t*>(buf);
    int consume_len = 0;
    int ret = 0;
    while (len > 0) {
        ret = ParsePkg(data, len, &consume_len);
        if (ret == -1)
            return -1;
        if (ret == 1) {
            if (proc)
                proc(ctx_, frame_, pkt_, private_data);
        } 
        data += consume_len;
        len -= consume_len;
    }
    return 0;
}

int FFmpegDemuxer::SendPacket()
{
    return ::avcodec_send_packet(ctx_, pkt_);
}

int FFmpegDemuxer::Decode()
{
    int ret = ::avcodec_receive_frame(ctx_, frame_);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
        return 0;
    else if (ret < 0)
        return -1;
    else
        return 1;
}

int FFmpegDemuxer::DecodeAll(DecodeFrameProc proc, void* private_data)
{
    int ret = 0;
    while (ret >= 0) {
        ret = ::avcodec_receive_frame(ctx_, frame_);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return 0;
        else if (ret < 0)
            return -1;
        else {
            if (proc)
                proc(ctx_, frame_, private_data);
        }
    }
    return 0;

#if 0
        snprintf(buf, sizeof(buf), "%s-%d", filename, dec_ctx->frame_number);
        //pgm_save(frame->data[0], frame->linesize[0], frame->width, frame->height, buf);

        if (frame->key_frame == 1) {
            //pgm_save(frame->data[0], frame->linesize[0], frame->width, frame->height, buf);
            printf("ctx_fnumber: %d %d %d    [%d %d %d]\n", dec_ctx->frame_number, frame->key_frame, frame->pts
                , frame->linesize[0], frame->width, frame->height
            );
        }
        std::cout << "pic_type: " << utils::picture_type(frame->pict_type)
            << " key_frame: " << frame->key_frame
            << " pic_frame_num: " << frame->coded_picture_number
            << " frame_number: " << dec_ctx->frame_number
            << " pts: " << frame->pts
            << " [" << frame->width << "," << frame->height
            << "\n";
#endif
}


