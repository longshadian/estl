#include <iostream>
#include <sstream>

#include "VideoForward.h"

int TestVideoForward()
{
    std::string rtsp_uri = "rtsp://192.168.1.95:8554/123.264";
    std::string rtmp_uri = "rtmp://192.168.1.95:31935/videotest/test";

    VideoForward vf;
    if (!vf.Init(rtsp_uri, rtmp_uri)) {
        std::cout << "init error\n";
        return 0;
    }
    vf.Loop();

    return 0;
}




