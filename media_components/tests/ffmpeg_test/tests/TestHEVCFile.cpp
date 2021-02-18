
#include <cstdio>
#include <string>
#include <array>
#include <vector>
#include <sstream>
#include <fstream>

#include <xffmpeg/xffmpeg.h>
#include "console_log.h"

static std::string g_save_file_path 
    = "/home/bolan/works/vsremote/media_components/out/build/Linux-GCC-Debug-234/tests/ffmpeg_test/bin/1.265";

static std::fstream g_fstm;
static void SaveVideo(std::string fpath, const void* pdata, size_t len)
{
    if (!g_fstm.is_open()) {
        g_fstm.open(fpath, std::ios::binary | std::ios::out | std::ios::trunc);
    }
    g_fstm.write((const char*)pdata, len);
}

static int DoDecodeHEVC(xffmpeg::MemParser& demuxer)
{
    static int g_n = 0;
    std::ostringstream ostm;
    //logPrintInfo("--->save file: %d", demuxer.pkt_->size);
    SaveVideo(g_save_file_path, demuxer.pkt_->data, demuxer.pkt_->size);

    ostm.str("");
    xffmpeg::avpacket_to_string(demuxer.pkt_, ostm);
    //std::cout << "avpacket: " << ostm.str() << "\n";

    ostm.str("");
    xffmpeg::avctx_to_string(demuxer.ctx_, ostm);
    //std::cout << "avctx: s" << ostm.str() << "\n";

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
                std::string fname = "hevc_";
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

int TestHEVCFile()
{
    std::string fname = "/home/bolan/works/videos/airport2.265";
    FILE* f = std::fopen(fname.c_str(), "rb");
    if (!f) {
        logPrintWarn("xxxx open file %s failure", fname.c_str());
        return 0;
    }

    xffmpeg::MemParser demuxer{};
    demuxer.Init_H265();

    //std::vector<char> buf(1024 * 1024, '\0');
    std::vector<char> buf(1024 * 1000, '\0');
    int ecode = 0;
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
        xffmpeg::MicrosecondTimer tm{};
        logPrintInfo("read file %d", len);
        do {
            tm.Start();
            parse_result = demuxer.Parse();
            tm.Stop();
            if (parse_result < 0) {
                logPrintWarn("parse pkg failure");
                return -1;
            }
            //if (parse_result > 0) {
                logPrintInfo("parse pkg result: %d cost: %f", parse_result, tm.GetMilliseconds());
            //}

            if (parse_result == 0)
                break;

            static int xx = 0;
            //if (xx == 0) {
            {
                ++xx;
                std::ostringstream ostm{};
                xffmpeg::avctx_to_string(demuxer.ctx_, ostm);
                std::cout << "----------->: " << ostm.str() << "\n";
            }
            ecode = demuxer.PrepareDecode();
            if (ecode < 0) {
                char str[128] = {0};
                ::av_strerror(ecode, str, sizeof(str));
                logPrintWarn("avcodec_send_pac  ket ecode: %d  reason: %s", ecode, str);
                continue;
            }
            ecode = DoDecodeHEVC(demuxer);
            if (ecode < 0) {
                logPrintWarn("DoDecodeHEVC failure");
                //return -1;
                continue;
            }
        } while (parse_result > 0);
    }

    return 0;
}

