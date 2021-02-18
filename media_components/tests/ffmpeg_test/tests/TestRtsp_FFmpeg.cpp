#include <iostream>
#include <sstream>

#include "console_log.h"
#include <xffmpeg/xffmpeg.h>
#include "RtspPull.h"

int TestRtsp_FFMpeg()
{
    xffmpeg::RtspPull puller;
    return puller.Init();
}
