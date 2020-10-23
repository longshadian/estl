#include <iostream>
#include <sstream>

#include "NvDec9.h"

int main()
{
    // 拉取rtsp流数据，解码后保存图片
    //std::string url = "rtsp://192.168.1.95:8554/yintai.264";
    std::string url = "rtsp://192.168.1.95:8554/beijing.264";
    std::string pic_dir = "/home/bolan/works/pic";
    std::string out_video_file = "/home/bolan/works/pic/beijin.264";

    NvDec9 dec;
    int ecode = dec.Init(0, cudaVideoCodec_H264);
    if (ecode)
        return -1;
    ecode = dec.InitEncode(1920, 1080, NV_ENC_BUFFER_FORMAT_YV12, NvEncoderInitParam(), out_video_file);
    if (ecode)
        return -1;
    ecode = dec.StartPullRtspThread(url);
    if (ecode)
        return -1;
    dec.StartDecoder(pic_dir);

    return 0;
}

