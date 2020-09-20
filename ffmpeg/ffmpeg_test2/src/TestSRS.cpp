#include <iostream>
#include <string>

#include "FileReader.h"
#include "xlive555/SRSRtmp.h"

#include "console_log.h"

std::shared_ptr<SrsRtmp> g_rtmp_yt;
const std::string g_yt_name = "E:/resource/uniubi/123.264";

std::shared_ptr<SrsRtmp> g_rtmp_xn;
const std::string g_xn_name = "E:/resource/uniubi/xiaoen.264";


std::shared_ptr<FileReader_H264> g_fr;
uint64_t pts = 0;
static void PktCallback(const std::string& fname, uint8_t* p, int len)
{
    pts += 30;
    if (fname == g_yt_name) {
        int ret = g_rtmp_yt->SendH264(p, len, pts);
        CONSOLE_LOG_INFO << "send h264 g_rtmp_yt: " << ret;
    } else if (fname == g_xn_name) {
        int ret = g_rtmp_xn->SendH264(p, len, pts);
        CONSOLE_LOG_INFO << "send h264 g_rtmp_xn: " << ret;
    }
}

int TestSRS()
{
    CONSOLE_LOG_INFO << "aaa";
    int ret = 0;

    if (1)
    {
        std::string yt_rtmp_url = "rtmp://192.168.1.15:1935/videotest/yt";
        g_rtmp_yt = std::make_shared<SrsRtmp>(yt_rtmp_url);
        ret = g_rtmp_yt->Init();
        if (ret != 0) {
            std::cout << __FUNCTION__ << " rtmp init failed.";
            return -1;
        }
    }

    if (1)
    {
        std::string xn_rtmp_url = "rtmp://192.168.1.15:1935/videotest/xn";
        g_rtmp_xn = std::make_shared<SrsRtmp>(xn_rtmp_url);
        ret = g_rtmp_xn->Init();
        if (ret != 0) {
            std::cout << __FUNCTION__ << " rtmp init failed.";
            return -1;
        }
    }

    g_fr = std::make_shared<FileReader_H264>();
    std::vector<std::string> video_vec = 
    {
        "E:/resource/uniubi/123.264",
        "E:/resource/uniubi/xiaoen.264",
    };
    for (const auto& fname : video_vec) {
        ret = g_fr->CreateReader(fname);
        if (ret < 0) {
            std::cout << __FUNCSIG__ << " create error";
            return -1;
        }
    }
    g_fr->Init(&PktCallback);

    while (1) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    return  0;
}
