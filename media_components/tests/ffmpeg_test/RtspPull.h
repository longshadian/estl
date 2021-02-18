#pragma once

#include <xffmpeg/xffmpeg.h>

namespace xffmpeg
{

class RtspPull
{
public:
    RtspPull();
    ~RtspPull();

    int Init();

    AVFormatContext* ifmt_ctx;
    int* stream_mapping;
};

} // namespace xffmpeg
