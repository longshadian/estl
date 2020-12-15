#pragma once

#include <sstream>
#include <chrono>

#include "xffmpeg.h"

namespace utils
{

void avpacket_to_string(const AVPacket* pkt, std::ostringstream& ostm);
void avctx_to_string(const AVCodecContext* ctx, std::ostringstream& ostm);

std::string picture_type(int v);


struct MicrosecondTimer
{
    MicrosecondTimer() : tb_(), te_() {}

    void Start() { tb_ = std::chrono::steady_clock::now(); }
    void Stop() { te_ = std::chrono::steady_clock::now(); }
    std::int64_t Delta() const { return std::chrono::duration_cast<std::chrono::microseconds>(te_ - tb_).count(); }
    float GetMilliseconds() const { return static_cast<float>(Delta()) / 1000.f; }

    std::chrono::steady_clock::time_point tb_;
    std::chrono::steady_clock::time_point te_;
};



}


