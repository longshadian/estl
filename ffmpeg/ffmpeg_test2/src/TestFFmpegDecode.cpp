#include <iostream>
#include <sstream>
#include <functional>
#include <fstream>

#include "VideoForward.h"
#include "xffmpeg/xffmpeg.h"
#include "xffmpeg/FFmpegUtils.h"

#include "console_log.h"

static void PickupFrame(const VideoForward::RtspFrame& frame);

static void pgm_save(unsigned char* buf, int wrap, int xsize, int ysize, char* filename)
{
    FILE* f;
    int i;

    f = fopen(filename, "w");
    fprintf(f, "P5\n%d %d\n%d\n", xsize, ysize, 255);
    for (i = 0; i < ysize; i++)
        fwrite(buf + i * wrap, 1, xsize, f);
    fclose(f);
}

class VideoForwardDecode : public VideoForward
{
public:
    VideoForwardDecode()
    {
    }

    virtual ~VideoForwardDecode()
    {
    }

    virtual void FrameProc(unsigned char* buffer, unsigned int buffer_length, unsigned numTruncatedBytes, struct timeval presentationTime, unsigned durationInMicroseconds) override
    {
        if (frame_num_ % 100 == 0) {
            if (frame_num_ == 0) {
                last_tp_ = std::chrono::high_resolution_clock::now();
            }
            else {
                auto tnow = std::chrono::high_resolution_clock::now();
                int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(tnow - last_tp_).count();
                last_tp_ = tnow;
                fps_ = ms / 100;
                if (fps_ <= 0)
                    fps_ = 30;
            }
        }
        ++frame_num_;
        pts_ += fps_;
        //logPrintInfo("frame: %d fps: %d pts: %d", (int)frame_num_, fps_, pts_);

        RtspFrame frame;
        frame.buffer = buffer;
        frame.buffer_length = buffer_length;
        frame.durationInMicroseconds = durationInMicroseconds;
        frame.numTruncatedBytes = numTruncatedBytes;
        frame.presentationTime = presentationTime;
        frame.pts = pts_;
        PostToRtmpQueue(frame);
        PickupFrame(frame);
    }

};

class FFmpegDecode
{
public:
    FFmpegDecode()
    {
    }

    ~FFmpegDecode()
    {

    }

    void Post(const VideoForward::RtspFrame& frame)
    {
        auto buffer = std::make_shared<VideoFrame>();
        buffer->Append((const char*)frame.buffer, frame.buffer_length, frame.pts);
        {
            std::lock_guard<std::mutex> lk{mtx_};
            buffer_.push_back(buffer);
            cond_.notify_one();
        }
    }

    bool Init()
    {
        if (!InitFFmpeg()) {
            logPrintError("ffmpeg init error");
            return false;
        }

        //ofsm_.open("./capture.264", std::ios::app|std::ios::binary);

        running = true;
        std::thread temp(std::bind(&FFmpegDecode::XThreadRun, this));
        thread_ = std::move(temp);
        return true;
    }

    void XThreadRun()
    {
        VideoFramePtr ptr;
        std::chrono::seconds s{1};
        while (running) {
            {
                std::unique_lock<std::mutex> lk{mtx_};
                cond_.wait_for(lk, s, [this](){ return !buffer_.empty(); });
                if (!buffer_.empty()) {
                    ptr = buffer_.front();
                    buffer_.pop_front();
                }
            }
            if (ptr) {
                Decode(ptr);
                SaveFile(ptr);
            }
            ptr = nullptr;
        }
    }

    bool InitFFmpeg()
    {
        codec_ = avcodec_find_decoder(AV_CODEC_ID_H264);
        if (!codec_) {
            fprintf(stderr, "Codec not found\n");
            return false;
        }

        parser_ = av_parser_init(codec_->id);
        if (!parser_) {
            fprintf(stderr, "parser not found\n");
            return false;
        }

        ctx_ = avcodec_alloc_context3(codec_);
        if (!ctx_) {
            fprintf(stderr, "Could not allocate video codec context\n");
            return false;
        }

        /* For some codecs, such as msmpeg4 and mpeg4, width and height
           MUST be initialized there because this information is not
           available in the bitstream. */

           /* open it */
        if (avcodec_open2(ctx_, codec_, NULL) < 0) {
            fprintf(stderr, "Could not open codec\n");
            return false;
        }

        frame_ = av_frame_alloc();
        if (!frame_) {
            fprintf(stderr, "Could not allocate video frame\n");
            return false;
        }
        pkt_ = av_packet_alloc();
        if (!pkt_)
            return false;
        return true;
    }

    void Decode(const VideoFramePtr& ptr)
    {
        find_iframe_ = true;
        if (!find_iframe_) {
            const char* p = ptr->Data();
            p += 4;
            char nut = (*p) & 0x1f;
            if (*p == 0x65 || nut == 1) {
                find_iframe_ = true;
            }
        }
        if (find_iframe_) {
            inbuf_.resize(ptr->Size() + AV_INPUT_BUFFER_PADDING_SIZE);
            std::copy(ptr->Data(), ptr->Data() + ptr->Size(), inbuf_.data());
            Decode1(inbuf_.data(), ptr->Size());
        }
    }

    void SaveFile(const VideoFramePtr& ptr)
    {
        if (0 && ofsm_.is_open()) {
            ofsm_.write(ptr->Data(), ptr->Size());
            ofsm_.flush();
        }
    }

    void Decode1(const uint8_t* buf, size_t data_size)
    {
        /* use the parser to split the data into frames */
        const uint8_t* data = buf;
        int ret = 0;
        int cnt = 0;
        while (data_size > 0) {
            ++cnt;
            ret = av_parser_parse2(parser_, ctx_, &pkt_->data, &pkt_->size,
                data, data_size, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
            if (ret < 0) {
                fprintf(stderr, "Error while parsing\n");
                exit(1);
            }
            data += ret;
            data_size -= ret;
            if (pkt_->size) {
                //std::ostringstream ostm;
                //utils::avpacket_to_string(pkt, ostm);
                //std::cout << ostm.str();

                Decode2(ctx_, frame_, pkt_);
                //exit(1);
            }
        }
    }

    static void Decode2(AVCodecContext* dec_ctx, AVFrame* frame, AVPacket* pkt)
    {
        int ret;

        ret = avcodec_send_packet(dec_ctx, pkt);
        if (ret < 0) {
            fprintf(stderr, "Error sending a packet for decoding\n");
            exit(1);
        }

        //std::ostringstream ostm;
        //utils::avctx_to_string(dec_ctx, ostm);
        //std::cout << ostm.str();

        while (ret >= 0) {
            //AV_PIX_FMT_YUV420P;
            //avcodec_decode_video2();
            // av_read_frame();
            ret = avcodec_receive_frame(dec_ctx, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                return;
            else if (ret < 0) {
                fprintf(stderr, "Error during decoding\n");
                exit(1);
            }

            //printf("saving frame %3d\n", dec_ctx->frame_number);
            fflush(stdout);

            /* the picture is allocated by the decoder. no need to free it */
#if 1

            if (frame->key_frame == 1) {
                char buf[1024];
                //snprintf(buf, sizeof(buf), "%s-%d.pgm", "a", dec_ctx->frame_number);
                snprintf(buf, sizeof(buf), "%s-%d.jpg", "a", dec_ctx->frame_number);
                ffmpeg_util_save_jpeg(frame, buf);
                //pgm_save(frame->data[0], frame->linesize[0], frame->width, frame->height, buf);
                printf("ctx_fnumber: %d %d %d    [%d %d %d]\n", dec_ctx->frame_number, frame->key_frame, frame->pts
                    , frame->linesize[0], frame->width, frame->height
                );
            }
#endif
#if 0
            std::cout << "pic_type: " << utils::picture_type(frame->pict_type)
                << " key_frame: " << frame->key_frame
                << " pic_frame_num: " << frame->coded_picture_number
                << " frame_number: " << dec_ctx->frame_number
                << " pts: " << frame->pts
                << " [" << frame->width << "," << frame->height
                << "\n";
#endif

        }
    }

    std::mutex mtx_{};
    std::condition_variable cond_{};
    std::list<VideoFramePtr> buffer_{};
    std::thread thread_{};
    bool running{};

    AVCodec* codec_{};
    AVCodecParserContext* parser_{};
    AVCodecContext* ctx_{};
    AVFrame* frame_{};
    std::vector<uint8_t> inbuf_{};
    //uint8_t inbuf_[10000 + AV_INPUT_BUFFER_PADDING_SIZE];
    //uint8_t* data;
    //size_t   data_size;
    AVPacket* pkt_{};
    bool find_iframe_{};

    bool save_file;
    std::string save_file_name;
    std::ofstream ofsm_;
};

FFmpegDecode g_decode;
void PickupFrame(const VideoForward::RtspFrame& frame) 
{
    g_decode.Post(frame);
}


int TestVideoForwardDecode()
{
    std::string rtsp_uri = "rtsp://192.168.1.95:8554/123.264";
    std::string rtmp_uri = "rtmp://192.168.1.95:31935/videotest/test";

    if (!g_decode.Init())
        return 0;

    VideoForwardDecode vf;
    if (!vf.Init(rtsp_uri, rtmp_uri)) {
        std::cout << "init error\n";
        return 0;
    }
    vf.Loop();

    return 0;
}



