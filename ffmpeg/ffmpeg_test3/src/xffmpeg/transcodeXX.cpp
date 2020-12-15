
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <sstream>
#include <iostream>
#include <ctime>
#include <chrono>

#include "xffmpeg/Utils.h"

extern "C"
{
#include <libavformat/avformat.h>
}

#define IO_BUFFER_SIZE 32768

static FILE* fp_open;

#if 1

/**
*在avformat_open_input()中会首次调用该回调函数，
*第二次一直到最后一次都是在avformat_find_stream_info()中循环调用，
*文件中的数据每次IO_BUFFER_SIZE字节读入到内存中，
*经过ffmpeg处理，所有数据被有序地逐帧存储到AVPacketList中。
*以上是缓存设为32KB的情况，缓存大小设置不同，调用机制也有所不同。
*/
static int fill_iobuffer(void* opaque, uint8_t* buf, int buf_size)
{
    if (!feof(fp_open)) {
        int true_size = fread(buf, 1, buf_size, fp_open);
        return true_size;
    } else {
        return -1;
    }
}

#endif

#if 0
int convert(const char* in_filename_v, const char* out_filename)
{
    AVInputFormat* ifmt_v = NULL;
    AVOutputFormat* ofmt = NULL;
    AVFormatContext* ifmt_ctx_v = NULL, * ofmt_ctx = NULL;
    AVPacket pkt;
    int ret, i;
    int videoindex_v = -1, videoindex_out = -1;
    int frame_index = 0;
    int64_t cur_pts_v = 0;

    //av_register_all();

    fp_open = fopen(in_filename_v, "rb+");
    ifmt_ctx_v = avformat_alloc_context();
    unsigned char* iobuffer = (unsigned char*)av_malloc(IO_BUFFER_SIZE);
    AVIOContext* avio = avio_alloc_context(iobuffer, IO_BUFFER_SIZE, 0, NULL, fill_iobuffer, NULL, NULL);
    ifmt_ctx_v->pb = avio;

    ifmt_v = av_find_input_format("h264");
    if ((ret = avformat_open_input(&ifmt_ctx_v, "nothing", ifmt_v, NULL)) < 0) {
        printf("Could not open input file.");
        //goto end;
        return 0;
    }

    if ((ret = avformat_find_stream_info(ifmt_ctx_v, 0)) < 0) {
        printf("Failed to retrieve input stream information");
        //goto end;
        return 0;
    }

    printf("===========Input Information==========\n");
    av_dump_format(ifmt_ctx_v, 0, in_filename_v, 0);
    printf("======================================\n");
    avformat_alloc_output_context2(&ofmt_ctx, NULL, NULL, out_filename);
    if (!ofmt_ctx) {
        printf("Could not create output context\n");
        ret = AVERROR_UNKNOWN;
        //goto end;
        return 0;
    }
    ofmt = ofmt_ctx->oformat;

    AVStream* in_stream = ifmt_ctx_v->streams[0];
    AVStream* out_stream = avformat_new_stream(ofmt_ctx, NULL);
    videoindex_v = 0;
    if (!out_stream) {
        printf("Failed allocating output stream\n");
        ret = AVERROR_UNKNOWN;
        //goto end;
        return 0;
    }

    videoindex_out = out_stream->index;
    //Copy the settings of AVCodecContext
    if (avcodec_parameters_copy(out_stream->codecpar, in_stream->codecpar) < 0) {
        printf("Failed to copy context from input to output stream codec context\n");
        //goto end;
        return 0;
    }

    out_stream->codecpar->codec_tag = 0;
    /* Some formats want stream headers to be separate. */
    /*
    if (ofmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        out_stream->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;
        */

    printf("==========Output Information==========\n");
    av_dump_format(ofmt_ctx, 0, out_filename, 1);
    printf("======================================\n");
    //Open output file
    if (!(ofmt->flags & AVFMT_NOFILE)) {
        if (avio_open(&ofmt_ctx->pb, out_filename, AVIO_FLAG_WRITE) < 0) {
            printf("Could not open output file '%s'", out_filename);
            //goto end;
            return 0;
        }
    }
    //Write file header
    if (avformat_write_header(ofmt_ctx, NULL) < 0) {
        printf("Error occurred when opening output file\n");
        //goto end;
        return 0;
    }

    while (1) {
        AVFormatContext* ifmt_ctx;
        int stream_index = 0;
        AVStream* in_stream, * out_stream;

        //Get an AVPacket
        ifmt_ctx = ifmt_ctx_v;
        stream_index = videoindex_out;

        if (av_read_frame(ifmt_ctx, &pkt) >= 0) {
            do {
                in_stream = ifmt_ctx->streams[pkt.stream_index];
                out_stream = ofmt_ctx->streams[stream_index];

                if (pkt.stream_index == videoindex_v) {
                    //FIX：No PTS (Example: Raw H.264)
                    //Simple Write PTS
                    if (pkt.pts == AV_NOPTS_VALUE) {
                        //Write PTS
                        AVRational time_base1 = in_stream->time_base;
                        //Duration between 2 frames (μs)
                        int64_t calc_duration = (double)AV_TIME_BASE / av_q2d(in_stream->r_frame_rate);
                        //Parameters
                        pkt.pts = (double)(frame_index * calc_duration) / (double)(av_q2d(time_base1) * AV_TIME_BASE);
                        pkt.dts = pkt.pts;
                        pkt.duration = (double)calc_duration / (double)(av_q2d(time_base1) * AV_TIME_BASE);
                        frame_index++;
                    }
                    cur_pts_v = pkt.pts;
                    break;
                }
            } while (av_read_frame(ifmt_ctx, &pkt) >= 0);
        }
        else {
            break;
        }

        //Convert PTS/DTS
        pkt.pts = av_rescale_q_rnd(pkt.pts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
        pkt.dts = av_rescale_q_rnd(pkt.dts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
        pkt.duration = av_rescale_q(pkt.duration, in_stream->time_base, out_stream->time_base);
        pkt.pos = -1;
        pkt.stream_index = stream_index;

        printf("Write 1 Packet. size:%5d\tpts:%lld\n", pkt.size, pkt.pts);
        //Write
        if (av_interleaved_write_frame(ofmt_ctx, &pkt) < 0) {
            printf("Error muxing packet\n");
            break;
        }
        //av_free_packet(&pkt);

    }

    //Write file trailer
    av_write_trailer(ofmt_ctx);

end:
    avformat_close_input(&ifmt_ctx_v);
    /* close output */
    if (ofmt_ctx && !(ofmt->flags & AVFMT_NOFILE))
        avio_close(ofmt_ctx->pb);
    avformat_free_context(ofmt_ctx);
    fclose(fp_open);
    if (ret < 0 && ret != AVERROR_EOF) {
        printf("Error occurred.\n");
        return -1;
    }
    return 0;
}

#endif

#if 0

struct output_stream
{

};

int Init(const char* out_filename)
{
    int ivideo_idx = 0;
    int ovideo_index = 0;
    int ret = 0;
    AVFormatContext* ifmt_ctx = NULL;
    AVIOContext* io_ctx = NULL;
    AVInputFormat* ifmt = NULL;

    AVFormatContext* ofmt_ctx = NULL;
    AVOutputFormat* ofmt = NULL;

    AVStream* istm = NULL;
    AVStream* ostm = NULL;

    // 创建输入流
    ifmt_ctx = avformat_alloc_context();
    unsigned char* iobuffer = (unsigned char*)av_malloc(IO_BUFFER_SIZE);
    io_ctx = avio_alloc_context(iobuffer, IO_BUFFER_SIZE, 0, NULL, fill_iobuffer, NULL, NULL);
    ifmt_ctx->pb = io_ctx;
    ifmt = av_find_input_format("h264");
    if ((ret = avformat_open_input(&ifmt_ctx, "nothing", ifmt, NULL)) < 0) {
        printf("Could not open input file.");
        //goto end;
        return 0;
    }
    if ((ret = avformat_find_stream_info(ifmt_ctx, 0)) < 0) {
        printf("Failed to retrieve input stream information");
        //goto end;
        return 0;
    }
    printf("===========Input Information==========\n");
    //av_dump_format(ifmt_ctx, 0, in_filename_v, 0);
    printf("======================================\n");


    // 创建输出流
    avformat_alloc_output_context2(&ofmt_ctx, NULL, NULL, out_filename);
    if (!ofmt_ctx) {
        printf("Could not create output context\n");
        ret = AVERROR_UNKNOWN;
        //goto end;
        return 0;
    }
    ofmt = ofmt_ctx->oformat;
    istm = ifmt_ctx->streams[0];
    ostm = avformat_new_stream(ofmt_ctx, NULL);
    ovideo_index = 0;
    if (!ostm) {
        printf("Failed allocating output stream\n");
        ret = AVERROR_UNKNOWN;
        //goto end;
        return 0;
    }
    ovideo_index = ostm->index;
    //Copy the settings of AVCodecContext
    if (avcodec_parameters_copy(istm->codecpar, ostm->codecpar) < 0) {
        printf("Failed to copy context from input to output stream codec context\n");
        //goto end;
        return 0;
    }
    ostm->codecpar->codec_tag = 0;
    /* Some formats want stream headers to be separate. */
    /*
    if (ofmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        out_stream->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;
        */
    printf("==========Output Information==========\n");
    //av_dump_format(ofmt_ctx, 0, out_filename, 1);
    printf("======================================\n");

    // 打开输出文件
    if (!(ofmt->flags & AVFMT_NOFILE)) {
        if (avio_open(&ofmt_ctx->pb, out_filename, AVIO_FLAG_WRITE) < 0) {
            printf("Could not open output file '%s'", out_filename);
            //goto end;
            return 0;
        }
    }
    // 写入文件头
    if (avformat_write_header(ofmt_ctx, NULL) < 0) {
        printf("Error occurred when opening output file\n");
        //goto end;
        return 0;
    }
}

void Write()
{
    int ivideo_idx = 0;
    int ovideo_index = 0;
    int ret = 0;
    AVFormatContext* ifmt_ctx = NULL;
    AVIOContext* io_ctx = NULL;
    AVInputFormat* ifmt = NULL;

    AVFormatContext* ofmt_ctx = NULL;
    AVOutputFormat* ofmt = NULL;

    AVStream* istm = NULL;
    AVStream* ostm = NULL;



    while (1) {
        AVFormatContext* ifmt_ctx;
        int stream_index = 0;
        AVStream* in_stream, * out_stream;

        //Get an AVPacket
        ifmt_ctx = ifmt_ctx_v;
        stream_index = videoindex_out;

        if (av_read_frame(ifmt_ctx, &pkt) >= 0) {
            do {
                in_stream = ifmt_ctx->streams[pkt.stream_index];
                out_stream = ofmt_ctx->streams[stream_index];

                if (pkt.stream_index == videoindex_v) {
                    //FIX：No PTS (Example: Raw H.264)
                    //Simple Write PTS
                    if (pkt.pts == AV_NOPTS_VALUE) {
                        //Write PTS
                        AVRational time_base1 = in_stream->time_base;
                        //Duration between 2 frames (μs)
                        int64_t calc_duration = (double)AV_TIME_BASE / av_q2d(in_stream->r_frame_rate);
                        //Parameters
                        pkt.pts = (double)(frame_index * calc_duration) / (double)(av_q2d(time_base1) * AV_TIME_BASE);
                        pkt.dts = pkt.pts;
                        pkt.duration = (double)calc_duration / (double)(av_q2d(time_base1) * AV_TIME_BASE);
                        frame_index++;
                    }
                    cur_pts_v = pkt.pts;
                    break;
                }
            } while (av_read_frame(ifmt_ctx, &pkt) >= 0);
        }
        else {
            break;
        }

        //Convert PTS/DTS
        pkt.pts = av_rescale_q_rnd(pkt.pts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
        pkt.dts = av_rescale_q_rnd(pkt.dts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
        pkt.duration = av_rescale_q(pkt.duration, in_stream->time_base, out_stream->time_base);
        pkt.pos = -1;
        pkt.stream_index = stream_index;

        printf("Write 1 Packet. size:%5d\tpts:%lld\n", pkt.size, pkt.pts);
        //Write
        if (av_interleaved_write_frame(ofmt_ctx, &pkt) < 0) {
            printf("Error muxing packet\n");
            break;
        }
        //av_free_packet(&pkt);

    }
}

void convert(const char* in_filename_v, const char* out_filename)
{
    AVOutputFormat* ofmt = NULL;
    AVFormatContext* ifmt_ctx_v = NULL, * ofmt_ctx = NULL;
    AVPacket pkt;
    int ret, i;
    int videoindex_v = -1, videoindex_out = -1;
    int frame_index = 0;
    int64_t cur_pts_v = 0;

    //av_register_all();

    fp_open = fopen(in_filename_v, "rb+");



    //Write file trailer
    av_write_trailer(ofmt_ctx);

end:
    avformat_close_input(&ifmt_ctx_v);
    /* close output */
    if (ofmt_ctx && !(ofmt->flags & AVFMT_NOFILE))
        avio_close(ofmt_ctx->pb);
    avformat_free_context(ofmt_ctx);
    fclose(fp_open);
    if (ret < 0 && ret != AVERROR_EOF) {
        printf("Error occurred.\n");
        return -1;
    }
    return 0;
}

#endif

#if 0
static
int Fun()
{
    int ivideo_idx = 0;
    int ret = 0;
    AVFormatContext* ifmt_ctx = NULL;
    AVIOContext* io_ctx = NULL;
    AVInputFormat* ifmt = NULL;

    AVFormatContext* ofmt_ctx = NULL;
    AVOutputFormat* ofmt = NULL;

    AVStream* istm = NULL;
    AVStream* ostm = NULL;

    const char* output_fname = "xx.mp4";

    // 创建输出流
    avformat_alloc_output_context2(&ofmt_ctx, NULL, NULL, output_fname);
    if (!ofmt_ctx) {
        printf("Could not create output context\n");
        ret = AVERROR_UNKNOWN;
        //goto end;
        return 0;
    }
    ofmt = ofmt_ctx->oformat;
    ostm = avformat_new_stream(ofmt_ctx, NULL);
    if (!ostm) {
        printf("Failed allocating output stream\n");
        ret = AVERROR_UNKNOWN;
        //goto end;
        return 0;
    }

    const AVCodec* encoder = avcodec_find_encoder(AV_CODEC_ID_MPEG4);
    if (!encoder) {
        printf("avcodec_find_encoder error\n");
        return 0;
    }
    AVCodecContext* encodec_ctx = avcodec_alloc_context3(encoder);
    if (!encodec_ctx) {
        printf("avcodec_alloc_context3 error\n");
        return 0;
    }
    encodec_ctx->height = 1280;
    encodec_ctx->width = 720;
    encodec_ctx->sample_aspect_ratio = {1, 25};
    if (encoder->pix_fmts)
        encodec_ctx->pix_fmt = encoder->pix_fmts[0];
    else
        encodec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    encodec_ctx->time_base = {1, 25};

    if (ofmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        encodec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;


    ret = avcodec_open2(encodec_ctx, encoder, NULL);
    if (ret < 0) {
        printf("avcodec_open2 error\n");
        return 0;
    }
    ostm->time_base = {1, 25};

    //ostm->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    if ((ret = avcodec_parameters_from_context(ostm->codecpar, encodec_ctx)) < 0) {
        printf("avcodec_parameters_from_context %d\n", ret);
        return 0;
    }

    //Open output file
    if (!(ofmt->flags & AVFMT_NOFILE)) {
        if (avio_open(&ofmt_ctx->pb, output_fname, AVIO_FLAG_WRITE) < 0) {
            printf("Could not open output file '%s'", "test.mp4");
            //goto end;
            return 0;
        }
    }
    //Write file header
    if ((ret = avformat_write_header(ofmt_ctx, NULL)) < 0) {
        printf("Error occurred when opening output file %d\n", ret);
        //goto end;
        return 0;
    }

    AVPacket* pkt = av_packet_alloc();
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    AVCodecParserContext* parser = av_parser_init(codec->id);
    if (!parser) {
        fprintf(stderr, "parser not found\n");
        return 0;
    }

    AVCodecContext* c = avcodec_alloc_context3(codec);
    if (!c) {
        fprintf(stderr, "Could not allocate video codec context\n");
        return 0;
    }
    if (avcodec_open2(c, codec, NULL) < 0) {
        fprintf(stderr, "Could not open codec\n");
        return 0;
    }

    const char* fname = "a.264";
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "Could not open %s\n", fname);
        exit(1);
    }

    AVFrame* frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Could not allocate video frame\n");
        exit(1);
    }

    size_t data_size;
    uint8_t* data;

#define INBUF_SIZE 1000
    uint8_t inbuf[INBUF_SIZE + AV_INPUT_BUFFER_PADDING_SIZE];

    while (!feof(f)) {
        /* read raw data from the input file */
        data_size = fread(inbuf, 1, INBUF_SIZE, f);
        if (!data_size)
            break;
        data = inbuf;
        while (data_size > 0) {
            ret = av_parser_parse2(parser, c, &pkt->data, &pkt->size,
                data, data_size, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
            if (ret < 0) {
                fprintf(stderr, "Error while parsing\n");
                exit(1);
            }
            data += ret;
            data_size -= ret;
            if (pkt->size) {
                /*
                //Convert PTS/DTS
                pkt.pts = av_rescale_q_rnd(pkt.pts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
                pkt.dts = av_rescale_q_rnd(pkt.dts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
                pkt.duration = av_rescale_q(pkt.duration, in_stream->time_base, out_stream->time_base);
                pkt.pos = -1;
                pkt.stream_index = stream_index;
                */
                std::ostringstream str_ostm;
                utils::avpacket_to_string(pkt, str_ostm);
                //printf("Write 1 Packet. size:%5d\tpts:%lld\n", pkt->size, pkt->pts);
                std::cout << str_ostm.str() << "\n";
                /*
                static int fps = 30;
                fps += 15;
                static int cnt = 0;
                pkt->pts = fps;
                */

                //Write
                #if 0
                if (av_interleaved_write_frame(ofmt_ctx, pkt) < 0) {
                    printf("Error muxing packet\n");
                    break;
                }
                #endif

                if (av_write_frame(ofmt_ctx, pkt) < 0) {
                    printf("Error muxing packet\n");
                    break;
                }
            }
        }
    }

    //Write file trailer
    av_write_trailer(ofmt_ctx);
    return 0;
}
#endif

struct XX
{
    const AVCodec* codec = NULL;
    AVFormatContext* fmt_ctx = NULL;
    AVCodecContext* codec_ctx = NULL;
    AVStream* stream = NULL;
    AVCodecParameters* parameters = NULL;
    AVPacket* pkt;
};

int xx_new(XX* x, int width, int height, const char* out_name)
{
    int ret = 0;
    AVCodecParameters* parameters = NULL;
    do {
        // 设置输出文件格式
        x->fmt_ctx = avformat_alloc_context();
        //x->fmt_ctx->oformat = av_guess_format("mjpeg", NULL, NULL);
        x->fmt_ctx->oformat = av_guess_format(NULL, out_name, NULL);

        // 创建并初始化输出AVIOContext
        ret = avio_open(&x->fmt_ctx->pb, out_name, AVIO_FLAG_WRITE);
        if (ret < 0) {
            break;
        }

        // 构建一个新stream
        x->stream = avformat_new_stream(x->fmt_ctx, NULL);
        if (!x->stream) {
            ret = -1;
            break;
        }

        parameters = x->stream->codecpar;
        parameters->codec_id = x->fmt_ctx->oformat->video_codec;
        parameters->codec_type = AVMEDIA_TYPE_VIDEO;
        parameters->format = AV_PIX_FMT_YUVJ420P;
        parameters->width = width;
        parameters->height = height;

        x->codec = avcodec_find_encoder(x->stream->codecpar->codec_id);
        if (!x->codec) {
            ret = -1;
            break;
        }
        x->codec_ctx = avcodec_alloc_context3(x->codec);
        if (!x->codec_ctx) {
            ret = -1;
            break;
        }

        ret = avcodec_parameters_to_context(x->codec_ctx, x->stream->codecpar);
        if (ret < 0) {
            //fprintf(stderr, "Failed to copy %s codec parameters to decoder context\n", av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
            break;
        }

        x->codec_ctx->height = height;
        x->codec_ctx->width = width;
        //x->codec_ctx->sample_aspect_ratio = { 1, 25 };
        if (x->codec->pix_fmts)
            x->codec_ctx->pix_fmt = x->codec->pix_fmts[0];
        else
            x->codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
            x->codec_ctx->time_base = { 1, 25 };

/*
        if (ofmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
            encodec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    */

        //codec_ctx->time_base = (AVRational){ 1, 25 };
        x->codec_ctx->time_base.num = 1;
        x->codec_ctx->time_base.den = 25;

        ret = avcodec_open2(x->codec_ctx, x->codec, NULL);
        if (ret < 0) {
            //printf("Could not open codec.");
            break;
        }

        ret = avformat_write_header(x->fmt_ctx, NULL);
        if (ret < 0) {
            //printf("write_header fail\n");
            break;
        }

        // Encode
        // 给AVPacket分配足够大的空间
        x->pkt = av_packet_alloc();
        /*
        ret = av_new_packet(&x->pkt, width * height * 3);
        if (ret != 0) {
            break;
        }
        */
        ret = 0;
    } while (0);

    if (ret < 0) {
    // TODO clean
    }
    return ret;
}

int xx_write(XX* x, const AVFrame* frame)
{
    int ret = 0;
    // 编码数据
    do {
        ret = avcodec_send_frame(x->codec_ctx, frame);
        if (ret < 0) {
            break;
        }

        // 得到编码后数据
        ret = avcodec_receive_packet(x->codec_ctx, x->pkt);
        if (ret < 0) {
            break;
        }
        //ret = av_write_frame(x->fmt_ctx, x->pkt);
        ret = av_interleaved_write_frame(x->fmt_ctx, x->pkt);
        if (ret < 0)
            break;
        ret = 0;
    } while (0);
    return ret;
}

int xx_close(XX* x)
{
    int ret = 0;
    //Write Trailer
    do {
        ret = av_write_trailer(x->fmt_ctx);
        if (ret < 0) {
            break;
        }
        ret = 0;
    } while (0);
    return ret;
}

void xx_destroy(XX* x)
{
    av_packet_unref(x->pkt);
    avio_close(x->fmt_ctx->pb);
    avcodec_free_context(&x->codec_ctx);
    avformat_free_context(x->fmt_ctx);
}

static int decode(AVCodecContext* dec_ctx, AVFrame* frame, AVPacket* pkt)
{
    int ret;
    ret = avcodec_send_packet(dec_ctx, pkt);
    if (ret < 0) {
        fprintf(stderr, "Error sending a packet for decoding\n");
        return ret;
    }

    while (ret >= 0) {
        ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return 0;
        else if (ret < 0) {
            fprintf(stderr, "Error during decoding\n");
            return ret;
        }
    }
    return 0;
}

static int Fun2()
{
    int ret = 0;

    AVPacket* pkt = av_packet_alloc();
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    AVCodecParserContext* parser = av_parser_init(codec->id);
    if (!parser) {
        fprintf(stderr, "parser not found\n");
        return 0;
    }

    AVCodecContext* c = avcodec_alloc_context3(codec);
    if (!c) {
        fprintf(stderr, "Could not allocate video codec context\n");
        return 0;
    }
    if (avcodec_open2(c, codec, NULL) < 0) {
        fprintf(stderr, "Could not open codec\n");
        return 0;
    }

    const char* fname = "a.264";
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "Could not open %s\n", fname);
        exit(1);
    }

    XX x;
    const char* output_fname = "xx.mp4";
    ret = xx_new(&x, 1280, 720, output_fname);
    if (ret < 0) {
        printf("xx_new error: %d\n", ret);
        return ret;
    }

    AVFrame* frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Could not allocate video frame\n");
        exit(1);
    }

    size_t data_size;
    uint8_t* data;

#define INBUF_SIZE 1000
    uint8_t inbuf[INBUF_SIZE + AV_INPUT_BUFFER_PADDING_SIZE];
    memset(inbuf + INBUF_SIZE, 0, AV_INPUT_BUFFER_PADDING_SIZE);

    std::time_t tbegin = time(nullptr);
    int frame_num_ = 0;
    std::chrono::high_resolution_clock::time_point last_tp_;
    int fps_ = 30;
    int pts_ = 500;
    while (!feof(f)) {
        /* read raw data from the input file */
        data_size = fread(inbuf, 1, INBUF_SIZE, f);
        if (!data_size)
            break;
        data = inbuf;
        while (data_size > 0) {
            ret = av_parser_parse2(parser, c, &pkt->data, &pkt->size,
                data, data_size, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
            if (ret < 0) {
                fprintf(stderr, "Error while parsing\n");
                exit(1);
            }
            data += ret;
            data_size -= ret;
            if (pkt->size) {
                std::ostringstream str_ostm;
                utils::avpacket_to_string(pkt, str_ostm);
                //printf("Write 1 Packet. size:%5d\tpts:%lld\n", pkt->size, pkt->pts);
                //std::cout << str_ostm.str() << "\n";

                if (frame_num_ % 100 == 0) {
                    if (frame_num_ == 0) {
                        last_tp_ = std::chrono::high_resolution_clock::now();
                    }
                    else {
                        auto tnow = std::chrono::high_resolution_clock::now();
                        int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(tnow - last_tp_).count();
                        last_tp_ = tnow;
                        fps_ = ms / 100;
                        if (fps_ <= 0)
                            fps_ = 30;
                    }
                }
                pts_ += fps_;
                ++frame_num_;
                //pkt->pts = pts_;
                //logPrintInfo("frame: %d fps: %d pts: %d", (int)frame_num_, fps_, pts_);

#if 0
                ret = decode(c, frame, pkt);
                if (ret < 0) {
                    printf("decode error %d\n", ret);
                    return -1;
                }
                ret = xx_write(&x, frame);
                if (ret < 0) {
                    printf("xx_write error %d\n", ret);
                    return -1;
                }
#endif

#if 0
                ret = avcodec_send_packet(c, pkt);
                if (ret < 0) {
                    fprintf(stderr, "Error sending a packet for decoding\n");
                    return ret;
                }

                while (ret >= 0) {
                    ret = avcodec_receive_frame(c, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                        break;
                    else if (ret < 0) {
                        fprintf(stderr, "Error during decoding\n");
                        return ret;
                    }
                    frame->pts = pts_;
                    if (frame->key_frame) {
                        int ret2 = xx_write(&x, frame);
                        if (ret2 < 0) {
                            printf("xx_write error %d\n", ret2);
                            return -1;
                        }
                    }
                }
#endif
                if (1) {
                    int ret2 = av_write_frame(x.fmt_ctx, pkt);
                    if (ret2 < 0) {
                        printf("Error muxing packet %d\n", ret2);
                        exit(1);
                    }
                }

                auto now = time(nullptr);
                if (now - tbegin > 10)
                    goto end;
            }
        }
    }

end:
    xx_close(&x);
    xx_destroy(&x);
    return 0;
}

int TestTranscodeXX()
{
    Fun2();
    return 0;
}
