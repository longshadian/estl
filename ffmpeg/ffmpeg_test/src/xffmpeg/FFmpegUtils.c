#include "FFmpegUtils.h"


int ffmpeg_util_save_jpeg(AVFrame* frame, const char* out_name)
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

/*
        std::cout << "pic_type: " << utils::picture_type(frame->pict_type)
            << " key_frame: " << frame->key_frame
            << " pic_frame_num: " << frame->coded_picture_number
            << " frame_number: " << dec_ctx->frame_number
            << " pts: " << frame->pts
            << " [" << frame->width << "," << frame->height
            << "\n";

*/
