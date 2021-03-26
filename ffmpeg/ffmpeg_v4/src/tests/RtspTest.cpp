#include "tests/RtspTest.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <array>
#include <vector>
#include <sstream>

#include "ffmpegx/FFMpegSdk.h"

int RtspTest()
{

    ffmpegx::RtspParam param{};
    //param.url = "rtsp://192.168.1.28:8554/xiaoen.264";
    param.url = "rtsp://10.106.104.3:554/openUrl/MSsUS9W";
    param.protocol_type = ffmpegx::TCP;

    std::unique_ptr<ffmpegx::FFMpegClient> client{ ffmpegx::CreateClient({}, &param) };
    assert(client->Init() == 0);
    client->Loop();

    return 0;
}

int RtmpTest()
{
    ffmpegx::RtmpParam param{};
    param.url = "rtmp://192.168.1.28:31935/live/abc";
    std::unique_ptr<ffmpegx::FFMpegClient> client{ ffmpegx::CreateClient({}, &param) };
    assert(client->Init() == 0);
    client->Loop();

    return 0;
}
