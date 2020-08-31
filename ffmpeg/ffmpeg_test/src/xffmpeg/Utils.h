#pragma once

#include <sstream>

#include "xffmpeg.h"

namespace utils
{

void avpacket_to_string(const AVPacket* pkt, std::ostringstream& ostm);
void avctx_to_string(const AVCodecContext* ctx, std::ostringstream& ostm);

std::string picture_type(int v);


}


