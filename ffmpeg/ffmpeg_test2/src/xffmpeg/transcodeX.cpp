
/**
*先将H.264文件读入内存，
*再输出封装格式文件。
*/

#include <cstdio>
#include <cstdint>
#include <cstddef>

extern "C"
{
#include <libavformat/avformat.h>
}

#define IO_BUFFER_SIZE 32768

static
FILE* fp_open;

#if 0

/**
*在avformat_open_input()中会首次调用该回调函数，
*第二次一直到最后一次都是在avformat_find_stream_info()中循环调用，
*文件中的数据每次IO_BUFFER_SIZE字节读入到内存中，
*经过ffmpeg处理，所有数据被有序地逐帧存储到AVPacketList中。
*以上是缓存设为32KB的情况，缓存大小设置不同，调用机制也有所不同。
*/
int fill_iobuffer(void* opaque, uint8_t* buf, int buf_size)
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

int TestTranscode2()
{
    //const char *in_filename_v = argv[1]; //Input file URL
    //const char *out_filename = argv[2]; //Output file URL

    const char* in_filename_v = "a.264"; //Input file URL
    const char* out_filename = "a.mp4"; //Output file URL
    //convert(in_filename_v, out_filename);
    convert(in_filename_v, out_filename);
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

