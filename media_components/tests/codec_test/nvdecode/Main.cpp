#include <iostream>
#include <sstream>
#include <memory>
#include <functional>

#include <xffmpeg/xffmpeg.h>

#include "NvDec.h"
#include "RtspClient.h"
#include "console_log.h"
#include "FFMpeg_RtspClient.h"
#include "file_utils.h"

static NvDec g_dec;

static xffmpeg::MemParser g_parser;

static file_utils g_fs;

static void FrameProc(
    unsigned char* buffer,
    unsigned int buffer_length,
    unsigned numTruncatedBytes,
    struct timeval presentationTime,
    unsigned durationInMicroseconds
)
{
#if 0
    g_parser.AppendRawData(buffer, buffer_length);
    int parse_result = 0;
    do {
        parse_result = g_parser.Parse();
        if (parse_result < 0) {
            logPrintWarn("parse result error: %d", parse_result);
            return;
        }

        if (parse_result == 0) {
            logPrintInfo("parse result >>>> 0");
        }

        if (parse_result > 0) {
            //assert(false);
            g_dec.FrameProc(g_parser.pkt_->data, g_parser.pkt_->size, numTruncatedBytes, presentationTime, durationInMicroseconds);
        }

    } while (parse_result > 0);
    //std::cout << "--------- buffer_length: " << buffer_length << "\n";
#endif

#if 0
    if (buffer_length > 10) {
        const uint8_t* pkt = (const uint8_t*)buffer + 1;
        printf("head: %d %d %d %d %d %d %d %d\n",
            pkt[0], pkt[1], pkt[2], pkt[3],
            pkt[4], pkt[5], pkt[6], pkt[7]
        );
    }
    g_dec.FrameProc(buffer + 1, buffer_length - 1, numTruncatedBytes, presentationTime, durationInMicroseconds);
#else
    g_dec.FrameProc(buffer, buffer_length, numTruncatedBytes, presentationTime, durationInMicroseconds);
#endif

#if 0
    {
        g_fs.write(buffer, buffer_length);
    }
#endif
}

#if 0
int TestPullRtsp()
{
    using namespace std::placeholders;

#if 0
    int ecode = g_dec.Init(0, cudaVideoCodec_H264, NvDec::MemoryType::Device);
    if (ecode)
        return -1;
    g_dec.StartDecoder_Background("./264");
    std::string url = "rtsp://192.168.1.95:8554/beijing.264";
#else
    int ecode = g_dec.Init(0, cudaVideoCodec_HEVC, NvDec::MemoryType::Device);
    if (ecode)
        return -1;
    g_dec.StartDecoder_Background("./265");
    std::string url = "rtsp://192.168.16.231/beijin.265";
#endif

    RtspPoller impl;
    RtspPollerParams params;
    params.url = url;
    params.frame_proc = std::bind(&::FrameProc, _1, _2, _3, _4, _5);
    if (!impl.Init(std::move(params))) {
        std::cout << "init error\n";
        return -1;
    }
    impl.Loop();
}
#endif

int TestPullRtsp()
{
    using namespace std::placeholders;

#if 0
    g_parser.Init_H264();
    int ecode = g_dec.Init(0, cudaVideoCodec_H264, NvDec::MemoryType::Device);
    if (ecode)
        return -1;
    g_dec.StartDecoder_Background("./264");
    std::string url = "rtsp://192.168.1.95:8554/beijing.264";
#else
    g_parser.Init_H265();
    int ecode = g_dec.Init(0, cudaVideoCodec_HEVC, NvDec::MemoryType::Device);
    if (ecode)
        return -1;
    g_dec.StartDecoder_Background("./265");
    std::string url = "rtsp://192.168.16.231/beijing.265";
#endif

#if 1
    RtspClient_Live555 client{};
    //RtspClient_FFMpeg client{};
    //RtspClient_File_HEVC client{};
    if (0) {
        //url = "/home/bolan/works/videos/airport2.265";
        //url = "/home/bolan/works/vsremote/media_components/out/build/Linux-GCC-Debug-234/tests/ffmpeg_test/bin/beijin_x.265";
        url = "/home/bolan/works/vsremote/media_components/out/build/Linux-GCC-Debug-234/beijing_dump.265";
    }

    g_fs.open("/home/bolan/works/vsremote/media_components/out/build/Linux-GCC-Debug-234/bin/save.265");
    //url = "rtsp://admin:test1234@192.168.70.33:554/cam/realmonitor?channel=1&subtype=0";
    url = "rtsp://192.168.16.237:8554/a.265";
    client.Init(url, FrameProc);
    client.Loop();
    //client.Loop_Decode();
    //client.Loop_3();
    //client.Loop_4();
#endif

#if 0
    //FFMpeg_File client; url = "/home/bolan/works/videos/10.265";
    FFMpeg_Rtsp_Decode client;
    //url = "rtsp://192.168.16.231/11.265";
    url = "rtsp://admin:test1234@192.168.70.33:554/cam/realmonitor?channel=1&subtype=0";
    //FFMpeg_Rtsp_Decode2 client;
    FFMpeg_Base* pbase = &client;
    pbase->Init(url, FrameProc);
    pbase->Loop();
#endif
    return 0;
}

int main()
{
    TestPullRtsp();
    return 0;
}


