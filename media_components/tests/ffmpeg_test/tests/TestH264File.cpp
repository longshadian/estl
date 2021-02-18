
#include <cstdio>
#include <string>
#include <array>
#include <vector>
#include <sstream>

#include <xffmpeg/xffmpeg.h>
#include "console_log.h"

static
int DoDecodeH264(xffmpeg::MemParser& demuxer)
{
    static int g_n = 0;
    std::ostringstream ostm;
    ostm.str("");
    xffmpeg::avpacket_to_string(demuxer.pkt_, ostm);
    //std::cout << "avpacket: " << ostm.str() << "\n";

    ostm.str("");
    xffmpeg::avctx_to_string(demuxer.ctx_, ostm);
    std::cout << "avctx: s" << ostm.str() << "\n";

    int decode_result = 0;
    do {
        decode_result = demuxer.Decode();
        if (decode_result < 0) {
            logPrintWarn("decode failure %d", decode_result);
            return -1;
        }
        if (decode_result > 0) {
            ++g_n;
#if 0
            auto* frame = demuxer.frame_;
            if (frame->key_frame == 1 || 1) {
                std::string fname = "xxx-";
                fname += std::to_string(g_n);
                fname += ".jpeg";
                xffmpeg::ffmpeg_save_jpeg(frame, fname.c_str());
            }
            std::cout << "pic_type: " << xffmpeg::picture_type(frame->pict_type)
                << " key_frame: " << frame->key_frame
                << " pic_frame_num: " << frame->coded_picture_number
                << " frame_number: " << demuxer.ctx_->frame_number
                << " pts: " << frame->pts
                << " [" << frame->width << "," << frame->height
                << "\n";
#endif
            if (g_n > 100 && true) {
                std::cout << "exit!!!" << std::endl;
                exit(0);
            }
        }
    } while (decode_result > 0);
    return 0;
}

int TestH264File()
{
    std::string fname = "/home/bolan/works/videos/yintai.264";
    FILE* f = std::fopen(fname.c_str(), "rb");
    if (!f) {
        logPrintWarn("xxxx open file %s failure", fname.c_str());
        return 0;
    }

    xffmpeg::MemParser demuxer{};
    demuxer.Init_H264();

    //std::vector<char> buf(1024 * 1024, '\0');
    std::vector<char> buf(1024, '\0');

    int ecode{};
    while (1) {
        int n = fread(buf.data(), 1, buf.size(), f);
        if (n == 0) {
            logPrintWarn("fread file n == 0");
            break;
        }
        
        auto* pbuf = buf.data();
        int len = (int)buf.size();
        int parse_result = 0;
        demuxer.AppendRawData(pbuf, len);
        do {
            parse_result = demuxer.Parse();
            if (parse_result < 0) {
                logPrintWarn("parse pkg failure");
                return -1;
            }
            if (parse_result > 0) {
                logPrintInfo("parse pkg result: %d", parse_result);
            }

            if (parse_result == 0)
                break;

            ecode = demuxer.PrepareDecode();
            if (ecode < 0) {
                char str[128] = {0};
                ::av_strerror(ecode, str, sizeof(str));
                logPrintWarn("avcodec_send_pac  ket ecode: %d  reason: %s", ecode, str);
                continue;
            }
            ecode = DoDecodeH264(demuxer);
            if (ecode < 0) {
                logPrintWarn("DoDecodeH264 failure");
                return -1;
            }
        } while (parse_result > 0);
    }

    return 0;
}

