#include <iostream>
#include <sstream>

#include "VideoForward.h"

int main()
{
    // 拉取rtsp流数据，转发至rtmp服务
    std::string rtsp_uri = "rtsp://192.168.1.95:8554/beijing.264";
    std::string rtmp_uri = "rtmp://192.168.16.234:1935/myapp/beijing.264";
    VideoForward vf;
    if (!vf.Init(rtsp_uri, rtmp_uri)) {
        std::cout << "init error\n";
        return 0;
    }
    vf.Loop();

    return 0;
}
