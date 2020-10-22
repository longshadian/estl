#include <iostream>
#include <sstream>

#include "NvDec.h"

int main()
{
    // 拉取rtsp流数据，解码后保存图片
    //std::string url = "rtsp://192.168.1.95:8554/yintai.264";
    std::string url = "rtsp://192.168.1.95:8554/beijing.264";
    std::string pic_dir = "/home/bolan/works/pic";

    NvDec dec;
    int ecode = dec.Init(0, cudaVideoCodec_H264);
    if (ecode)
        return -1;
    ecode = dec.StartPullRtspThread(url);
    if (ecode)
        return -1;
    dec.StartDecoder(pic_dir);

    return 0;
}

