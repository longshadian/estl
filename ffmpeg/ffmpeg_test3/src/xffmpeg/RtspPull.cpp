#include "RtspPull.h"

#include <cassert>
#include <iostream>
#include <sstream>
#include "file_utils.h"
#include "Utils.h"

namespace xffmpeg
{

RtspPull::RtspPull()
{
}

RtspPull::~RtspPull()
{
}

int RtspPull::Init()
{
    AVFormatContext* ifmt_ctx = NULL;
    AVPacket pkt;
    const char* out_filename = nullptr;;
    int ret, i;
    int stream_index = 0;
    int* stream_mapping = NULL;
    int stream_mapping_size = 0;
    file_utils fu{};
    assert(fu.open("D:/github/estl/ffmpeg/ffmpeg_test3/abc.h264"));

    std::string in_filename = "rtsp://192.168.16.231:8554/yintai.264";

    AVDictionary* options{};
    if (0) {
        ::av_dict_set(&options, "rtsp_transport", "tcp", 0);
    }

    // 1. 打开输入
    // 1.1 读取文件头，获取封装格式相关信息
    if ((ret = avformat_open_input(&ifmt_ctx, in_filename.c_str(), 0, &options)) < 0) {
        printf("Could not open input file '%s'", in_filename.c_str());
        goto end;
    }

    // 1.2 解码一段数据，获取流相关信息
    if ((ret = avformat_find_stream_info(ifmt_ctx, 0)) < 0) {
        printf("Failed to retrieve input stream information");
        goto end;
    }

    av_dump_format(ifmt_ctx, 0, in_filename.c_str(), 0);

    // 2. 打开输出
    // 2.1 分配输出ctx
    bool push_stream = true;
    char* ofmt_name = NULL;
    stream_mapping_size = ifmt_ctx->nb_streams;
    stream_mapping = (int*)av_mallocz_array(stream_mapping_size, sizeof(*stream_mapping));
    if (!stream_mapping) {
        ret = AVERROR(ENOMEM);
        goto end;
    }

    AVRational frame_rate;
    double duration;
    for (i = 0; i < ifmt_ctx->nb_streams; i++) {
        AVStream* in_stream = ifmt_ctx->streams[i];
        AVCodecParameters* in_codecpar = in_stream->codecpar;

        if (in_codecpar->codec_type != AVMEDIA_TYPE_AUDIO &&
            in_codecpar->codec_type != AVMEDIA_TYPE_VIDEO &&
            in_codecpar->codec_type != AVMEDIA_TYPE_SUBTITLE) {
            stream_mapping[i] = -1;
            continue;
        }

        if (push_stream && (in_codecpar->codec_type == AVMEDIA_TYPE_VIDEO)) {
            frame_rate = av_guess_frame_rate(ifmt_ctx, in_stream, NULL);
            AVRational r;
            r.den = frame_rate.den;
            r.num = frame_rate.num;
            duration = (frame_rate.num && frame_rate.den ? av_q2d(r) : 0);
        }

        stream_mapping[i] = stream_index++;
    }

    utils::MicrosecondTimer timer;
    while (1) {
        AVStream* in_stream;

        // 此处如果是从rtsp server获取的流，这个耗时就是每帧耗时。
        timer.Start();
        // 3.2 从输出流读取一个packet
        ret = av_read_frame(ifmt_ctx, &pkt);
        timer.Stop();
        std::cout << "cost: " << timer.GetMilliseconds();
        if (ret < 0) {
            break;
        }

        std::ostringstream ostm{};
        avpacket_to_string(&pkt, ostm);
        //std::cout << ostm.str() << "\n";

        in_stream = ifmt_ctx->streams[pkt.stream_index];
        if (pkt.stream_index >= stream_mapping_size ||
            stream_mapping[pkt.stream_index] < 0) {
            av_packet_unref(&pkt);
            continue;
        }

        int codec_type = in_stream->codecpar->codec_type;
        if (push_stream && (codec_type == AVMEDIA_TYPE_VIDEO)) {
            int64_t d = (int64_t)(duration * AV_TIME_BASE);
            std::cout << d << "\n";
            //av_usleep((int64_t)(duration * AV_TIME_BASE));
        }
        if (0) {
            if (pkt.size >= 4) {
                printf("head: %d %d %d %d\n", pkt.data[0], pkt.data[1], pkt.data[2], pkt.data[3]);
            }
            //for (int i = 0; i != headn; ++i) {
            //}
        }
        fu.write(pkt.data, pkt.size);

        pkt.stream_index = stream_mapping[pkt.stream_index];

        if (ret < 0) {
            printf("Error muxing packet\n");
            break;
        }
        av_packet_unref(&pkt);
    }

end:
    avformat_close_input(&ifmt_ctx);
    av_freep(&stream_mapping);

    if (ret < 0 && ret != AVERROR_EOF) {
        printf("Error occurred:\n");
        //printf("Error occurred: %s\n", (av_err2str(ret)));
        //av_make_error_string((char[AV_ERROR_MAX_STRING_SIZE]) { 0 }, AV_ERROR_MAX_STRING_SIZE, errnum)
        return 1;
    }

    return 0;
}

} // namespace xffmpeg


int TestRtspPull()
{
    xffmpeg::RtspPull puller;
    return puller.Init();
}


