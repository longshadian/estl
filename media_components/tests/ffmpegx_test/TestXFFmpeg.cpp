#include <cstdint>
#include <cstdio>
#include <array>
#include <cassert>

#include "xffmpeg/xffmpeg.h"
#include "console_log.h"

#define BUF_SIZE 100

#define XEXIT(v) assert(0)

static void decodePkt(AVCodecContext* dec_ctx, AVPacket* pkt, AVFrame* frame, void* pdata)
{
    const char* filename = (const char*)pdata;
    char buf[1024];
    //printf("saving frame %3d\n", dec_ctx->frame_number);
    fflush(stdout);
    /* the picture is allocated by the decoder. no need to
       free it */
    snprintf(buf, sizeof(buf), "%s-%d.jpeg", filename, dec_ctx->frame_number);
    //pgm_save(frame->data[0], frame->linesize[0], frame->width, frame->height, buf);
#if 1
        if (frame->key_frame == 1) {
            //pgm_save(frame->data[0], frame->linesize[0], frame->width, frame->height, buf);
            //printf("ctx_fnumber: %d %d %d    [%d %d %d]\n", dec_ctx->frame_number, frame->key_frame, frame->pts , frame->linesize[0], frame->width, frame->height );

            xffmpeg::ffmpeg_save_jpeg(frame, buf);
        }
#endif
    std::cout << "pic_type: " << xffmpeg::picture_type(frame->pict_type)
        << " key_frame: " << frame->key_frame
        << " pic_frame_num: " << frame->coded_picture_number
        << " frame_number: " << dec_ctx->frame_number
        << " pts: " << frame->pts
        << " [" << frame->width << "," << frame->height
        << "\n";
}


int TestXFFmpeg_MemoryDemuxer()

{
    const char* fname = "E:/resource/uniubi/123.264";
    std::array<uint8_t, BUF_SIZE> buf;

    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
    uint8_t* pVideo = nullptr;
    int ret = 0;

    xffmpeg::MemoryDemuxer demuxer{};
    if (demuxer.Init() != 0) {
        CONSOLE_PRINT_WARN("memory demuxer init error");
        return -1;
    }

    FILE* f = ::fopen(fname, "rb");
    if (!f) {
        CONSOLE_PRINT_WARN("open file error %s", fname);
        return -1;
    }
    size_t readn = 0;
    bool running = true;
    while (running) {
        do {
            ret = demuxer.Demux(&pVideo, &nVideoBytes);
            if (ret < 0) {
                CONSOLE_PRINT_WARN("Demux failuer");
                return -1;
            } else if (ret == 0) {
                readn = ::fread(buf.data(), 1, buf.size(), f);
                if (readn == 0) {
                    CONSOLE_PRINT_INFO("read file EOF!");
                    running = false;
                    break;
                }
                demuxer.data_ = buf.data();
                demuxer.data_size_ = readn;
            }
        } while (ret == 0);

        xffmpeg::decode(demuxer.ctx_, demuxer.pkt_, demuxer.frame_, &decodePkt, "testmm-");
    }
    xffmpeg::decode(demuxer.ctx_, demuxer.pkt_, demuxer.frame_, &decodePkt, "testmm-");
}

