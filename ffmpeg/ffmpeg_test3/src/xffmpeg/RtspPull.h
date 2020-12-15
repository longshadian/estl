#pragma once

#include "xffmpeg/xffmpeg.h"

namespace xffmpeg
{

class RtspPull
{
public:
    RtspPull();
    ~RtspPull();

    int Init();
};

} // namespace xffmpeg
