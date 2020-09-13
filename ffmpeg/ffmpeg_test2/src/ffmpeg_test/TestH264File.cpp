#include "xffmpeg/FFmpegDemuxer.h"

#include <cstdio>
#include <string>
#include <array>
#include <vector>
#include <sstream>

#include "console_log.h"
#include "xffmpeg/FFmpegUtils.h"
#include "xffmpeg/Utils.h"

int g_n = 0;


static
int DoDecode(FFmpegDemuxer& demuxer)
{
    int ret = 0;
    std::ostringstream ostm;
    ret = demuxer.SendPacket();
    if (ret == -1) {
        logPrintWarn("send packet failure %d", ret);
        return -1;
    }

    ostm.str("");
    utils::avpacket_to_string(demuxer.pkt_, ostm);
    std::cout << "avpacket: " << ostm.str() << "\n";

    ostm.str("");
    utils::avctx_to_string(demuxer.ctx_, ostm);
    std::cout << "avctx: s" << ostm.str() << "\n";

    do {
        ret = demuxer.Decode();
        if (ret < 0) {
            logPrintWarn("decode failure %d", ret);
            return -1;
        }
        if (ret > 0) {
#if 1
            ++g_n;
            auto* frame = demuxer.frame_;
            if (frame->key_frame == 1) {
                std::string fname = "xxx-";
                fname += std::to_string(g_n);
                fname += ".jpeg";
                ffmpeg_util_save_jpeg(frame, fname.c_str());
            }
            std::cout << "pic_type: " << utils::picture_type(frame->pict_type)
                << " key_frame: " << frame->key_frame
                << " pic_frame_num: " << frame->coded_picture_number
                << " frame_number: " << demuxer.ctx_->frame_number
                << " pts: " << frame->pts
                << " [" << frame->width << "," << frame->height
                << "\n";
#endif
            if (g_n > 100 && false) {
                std::cout << "exit!!!" << std::endl;
                exit(0);
            }
        }
    } while (ret > 0);
    return 0;
}

int TestH264File()
{
    std::string fname = "E:/resource/xiaoen.h264";
    FILE* f = std::fopen(fname.c_str(), "rb");
    if (!f) {
        logPrintWarn("xxxx open file %s failure", fname.c_str());
        return 0;
    }

    FFmpegDemuxer demuxer{};
    demuxer.Init();

    std::vector<char> buf(1024 * 1024, '\0');
    //std::array<char, 1024 * 4> buf ;
    int ret = 0;
    while (1) {
        int n = fread(buf.data(), 1, buf.size(), f);
        if (n == 0)
            break;
        
        auto* pbuf = buf.data();
        int len = (int)buf.size();
        int consume_len = 0;
        while (len > 0) {
            ret = demuxer.ParsePkg(pbuf, len, &consume_len);
            if (ret < 0) {
                logPrintWarn("parse pkg failure");
                return -1;
            }

            int rr = DoDecode(demuxer);
            if (rr < 0) {
                logPrintWarn("DoDecode failure");
                return -1;
            }
            pbuf += consume_len;
            len -= consume_len;
        }
    }

    return 0;
}

