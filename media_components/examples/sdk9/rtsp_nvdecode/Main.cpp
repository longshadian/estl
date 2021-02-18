#include <iostream>
#include <sstream>

#include "NvDec.h"

int main()
{
    //::setenv("CUDA_VISIBLE_DEVICES", "4", 0);
    // 拉取rtsp流数据，解码后保存图片
    NvDec dec{};
    int ecode{};
#if 0
    std::string url = "rtsp://192.168.1.95:8554/yintai.264";
    std::string pic_dir = "/home/bolan/works/pic";
    ecode = dec.Init(0, cudaVideoCodec_H264, NvDec::MemoryType::Device);
#else
    std::string url = "rtsp://192.168.16.231/airport2.265";
    std::string pic_dir = "/home/bolan/works/pic";
    ecode = dec.Init(0, cudaVideoCodec_HEVC, NvDec::MemoryType::Device);
#endif

    if (ecode)
        return -1;
    ecode = dec.StartPullRtspThread(url);
    if (ecode)
        return -1;
    dec.StartDecoder(pic_dir);

    return 0;
}

