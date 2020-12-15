#include "MyFFmpegDemuxer.h"

#include "console_log.h"

MyFFmpegDemuxer::MyFFmpegDemuxer()
    : const_codec_()
    , parser_()
    , ctx_()
    , pkt_()
    , frame_()

    , pktFiltered_()
    , bsfc_()
    , data_()
    , data_size_()
    , bMp4H264()
    , bMp4HEVC()
    , bMp4MPEG4()
{
}

MyFFmpegDemuxer::~MyFFmpegDemuxer()
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

int MyFFmpegDemuxer::Init(AVCodecID codec_id)
{
    // TODO true
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

    pktFiltered_ = av_packet_alloc();
    if (!pktFiltered_)
        return -1;

    // Initialize bitstream filter and its required resources
    if (bMp4H264) {
        const AVBitStreamFilter* bsf = av_bsf_get_by_name("h264_mp4toannexb");
        if (!bsf) {
            std::cout <<"FFmpeg error: " << __FILE__ << " " << __LINE__ << " " << "av_bsf_get_by_name() failed";
            return -1;
        }
        if (av_bsf_alloc(bsf, &bsfc_) != 0)
            return -1;
            // TODO
        //avcodec_parameters_copy(bsfc_->par_in, fmtc->streams[iVideoStream]->codecpar);
        if (av_bsf_init(bsfc_) != 0) {
            return -1;
        }
    }
    if (bMp4HEVC) {
        const AVBitStreamFilter* bsf = av_bsf_get_by_name("hevc_mp4toannexb");
        if (!bsf) {
            std::cout << "FFmpeg error: " << __FILE__ << " " << __LINE__ << " " << "av_bsf_get_by_name() failed";
            return -1;
        }
        if (av_bsf_alloc(bsf, &bsfc_) != 0)
            return -1;
        // TODO
        //avcodec_parameters_copy(bsfc_->par_in, fmtc->streams[iVideoStream]->codecpar);
        if (av_bsf_init(bsfc_) != 0) {
            return -1;
        }
    }
    return 0;
}

int MyFFmpegDemuxer::ParsePkg(const void* buf, int len, int* consume_len) 
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

int MyFFmpegDemuxer::ParsePkgAll(const void* buf, int len, ParsePkgProc proc, void* private_data)
{
    const uint8_t* data = reinterpret_cast<const uint8_t*>(buf);
    int data_size = len;
    int ret = 0;
    while (data_size > 0) {
        ret = ::av_parser_parse2(parser_, ctx_, &pkt_->data, &pkt_->size,
            data, data_size, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
        if (ret < 0) {
            return -1;
        }
        data += ret;
        data_size -= ret;
        if (pkt_->size) {
            //decode(c, frame, pkt, outfilename);
        }
    }
}

int MyFFmpegDemuxer::SendPacket()
{
    return ::avcodec_send_packet(ctx_, pkt_);
}

int MyFFmpegDemuxer::Decode()
{
    int ret = ::avcodec_receive_frame(ctx_, frame_);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
        return 0;
    else if (ret < 0)
        return -1;
    else
        return 1;
}

int MyFFmpegDemuxer::DecodeAll(DecodeFrameProc proc, void* private_data)
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

int MyFFmpegDemuxer::Demux(uint8_t** ppVideo, int* pnVideoBytes, int64_t* pts) 
{
    if (!ctx_) {
        return -1;
    }
    *pnVideoBytes = 0;
    /*
    if (pkt.data) {
        av_packet_unref(&pkt);
    }
    */

    bool has_pkg = false;
    int ret = 0;
    while (data_size_ > 0) {
        ret = ::av_parser_parse2(parser_, ctx_, &pkt_->data, &pkt_->size,
            data_, data_size_, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
        if (ret < 0) {
            return -1;
        }
        data_ += ret;
        data_size_ -= ret;
        if (pkt_->size) {
            has_pkg = true;
            break;
        }
    }
    if (!has_pkg)
        return 0;
    *ppVideo = pkt_->data;
    *pnVideoBytes = pkt_->size;
    return 1;
}


