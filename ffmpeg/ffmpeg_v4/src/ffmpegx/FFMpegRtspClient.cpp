#include "ffmpegx/FFMpegRtspClient.h"

#include <cassert>

#include "ffmpegx/FFMpegX.h"
#include "console_log.h"

#include "file_utils.h"

namespace ffmpegx
{

#define FFMPEGX_LOG_INFO logPrintInfo
#define FFMPEGX_LOG_WARN logPrintWarn

file_utils g_file;

FFMpegClient::FFMpegClient(FrameCallback cb)
    : ifmt_ctx_{}
    , options_{}
    , pkt_{}
    , running_{}
    , inited_{}
    , last_read_tp_{}
    , cb_{std::move(cb)}
    , read_timeout_{}
    , url_{}
{
}

FFMpegClient::~FFMpegClient()
{
    if (ifmt_ctx_)
        ::avformat_close_input(&ifmt_ctx_);
    if (options_)
        ::av_dict_free(&options_);
    if (pkt_)
        ::av_packet_free(&pkt_);
}

int FFMpegClient::Init()
{
    ifmt_ctx_ = ::avformat_alloc_context();
    ifmt_ctx_->interrupt_callback.callback = &FFMpegClient::CheckInterrupt;
    ifmt_ctx_->interrupt_callback.opaque = this;

    // 1. 打开输入
    // 1.1 读取文件头，获取封装格式相关信息
    int ecode = ::avformat_open_input(&ifmt_ctx_, url_.c_str(), 0, &options_);

    if (ecode < 0) {
        char buf[512] = { 0 };
        const char* pstr = ::av_make_error_string(buf, sizeof(buf), ecode);
        FFMPEGX_LOG_WARN("Could not open input url: %s %s", url_.c_str(), pstr);
        return ecode;
    }

    // 1.2 解码一段数据，获取流相关信息
    if ((ecode = ::avformat_find_stream_info(ifmt_ctx_, 0)) < 0) {
        FFMPEGX_LOG_WARN("Failed to retrieve input stream information");
        return ecode;
    }

    ::av_dump_format(ifmt_ctx_, 0, url_.c_str(), 0);
    for (unsigned i = 0; i < ifmt_ctx_->nb_streams; i++) {
        AVStream* in_stream = ifmt_ctx_->streams[i];
        AVCodecParameters* in_codecpar = in_stream->codecpar;
        std::ostringstream stsm{};
        AVCodecParameters_to_string(in_codecpar, stsm);
        FFMPEGX_LOG_INFO("params: %d %s", in_codecpar->codec_type, stsm.str().c_str());
        if (in_codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            FFMPEGX_LOG_INFO("i--------------------");
        }
    }

    inited_ = true;
    return 0;
}

int FFMpegClient::Loop()
{
    running_ = true;
    int ecode{};
    MicrosecondTimer timer;
    pkt_ = ::av_packet_alloc();
    while (running_) {
        // 此处如果是从rtsp server获取的流，这个耗时就是每帧耗时。
        timer.Start();
        // 3.2 从输出流读取一个packet
        ecode = ::av_read_frame(ifmt_ctx_, pkt_);
        timer.Stop();
        if (ecode < 0) {
            FFMPEGX_LOG_WARN("read av_read_frame failure, ecode: %d", ecode);
            break;
        }
        last_read_tp_ = Clock::now();
        
        FFMPEGX_LOG_INFO("stream index: %d   cost: %f",  static_cast<int>(pkt_->stream_index), timer.GetMilliseconds());
        OnReadPkt(pkt_);
        ::av_packet_unref(pkt_);
    }
    return 0;
}

int FFMpegClient::OnReadPkt(AVPacket* pkt)
{
    int idx = pkt->stream_index;
    AVStream* in_stream = ifmt_ctx_->streams[idx];
    AVCodecParameters* in_codecpar = in_stream->codecpar;
    if (in_codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
        g_file.write(pkt->data, pkt->size);

        std::ostringstream ostm{};
        AVPacket_to_string(pkt, ostm);
        FFMPEGX_LOG_INFO("stream_index: %d %d  %s", idx, in_codecpar->codec_id, ostm.str().c_str());

        if (pkt->size >= 10) {
            FFMPEGX_LOG_INFO("stream_index: %d head: \n%x %x %x %x %x \n%x %x %x %x %x\n%x %x %x %x %x \n%x %x %x %x %x\n",
                idx,
                pkt->data[0], pkt->data[1], pkt->data[2], pkt->data[3], pkt->data[4], pkt->data[5], pkt->data[6], pkt->data[7], pkt->data[8], pkt->data[9],
                pkt->data[10], pkt->data[11], pkt->data[12], pkt->data[13], pkt->data[14], pkt->data[15], pkt->data[16], pkt->data[17], pkt->data[18], pkt->data[19]
            );
        }
    }
    return 0;
}

int FFMpegClient::CheckInterrupt(void* ctx)
{
    FFMpegClient* pthis = static_cast<FFMpegClient*>(ctx);
    return pthis->CheckInterruptEx();
}

int FFMpegClient::CheckInterruptEx()
{
    if (!inited_)
        return 0;
    if (!running_)
        return 1;
    auto tnow = std::chrono::steady_clock::now();
    //FFMPEGX_LOG_WARN("-------------> timeout");
    return (tnow - last_read_tp_) > read_timeout_;
}

//=======================================================================
// 
//=======================================================================

#if 0
FFMpeg_Rtsp_Decode::FFMpeg_Rtsp_Decode()
    : Super{}
    , const_codec_{}
    , dec_ctx_{}
    , dec_frame_{}
{
}

FFMpeg_Rtsp_Decode::~FFMpeg_Rtsp_Decode()
{
    if (dec_frame_)
        ::av_frame_free(&dec_frame_);
}

int FFMpeg_Rtsp_Decode::Init(std::string url, Callback_t cb)
{
    if (Super::Init(url, cb) < 0)
        return -1;

    const_codec_ = ::avcodec_find_decoder(AV_CODEC_ID_H265);
    //const_codec_ = ::avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!const_codec_) {
        return -1;
    }
    dec_ctx_ = ::avcodec_alloc_context3(const_codec_);
    if (!dec_ctx_) {
        return -1;
    }

#if 1
    if (::avcodec_parameters_to_context(dec_ctx_, vid_stream_->codecpar) < 0) {
        return -1;
    }
#endif

    /* For some codecs, such as msmpeg4 and mpeg4, width and height
       MUST be initialized there because this information is not
       available in the bitstream. */
       /* open it */
    if (::avcodec_open2(dec_ctx_, const_codec_, NULL) < 0) {
        return -1;
    }
    dec_frame_ = ::av_frame_alloc();
    if (!dec_frame_) {
        return -1;
    }
    return 0;
}

int FFMpeg_Rtsp_Decode::OnReadPkt(AVPacket* pkt)
{
    if (pkt->size >= 10) {
        printf("stream_index: %d head: %d %d %d %d %d %d %d %d\n",
            stream_index_,
            pkt->data[0], pkt->data[1], pkt->data[2], pkt->data[3],
            pkt->data[4], pkt->data[5], pkt->data[6], pkt->data[7]
        );
    }
    //cb_(pkt->data, pkt->size, 0, {}, 0);
    if (1) {
        int ret = ::avcodec_send_packet(dec_ctx_, pkt);
        if (ret < 0) {
            printf("Error sending a packet for decoding\n");
            return -1;
        }
        while (ret >= 0) {
            ret = avcodec_receive_frame(dec_ctx_, dec_frame_);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                return 0;
            else if (ret < 0) {
                printf("Error during decoding\n");
                return -1;
            }
            std::ostringstream ostm{};
            xffmpeg::AVPacket_to_string(pkt, ostm);
            //std::cout << ostm.str() << "\n";
            decode_proc(dec_ctx_, pkt, dec_frame_);
        }
        return 0;
    }
}

void FFMpeg_Rtsp_Decode::decode_proc(AVCodecContext*, AVPacket*, AVFrame* frame)
{
    static int g_n = 0;
    ++g_n;
    if (g_n % 10 != 0)
        return;

    if (1) {
        // save yuv
        std::string fname = "ffmpeg_decode_";
        fname += std::to_string(g_n);
        fname += ".yuv";
        file_utils::write_file(fname.c_str(), frame->data[0], frame->width * frame->height);
        return;
    }

    if (frame->key_frame == 1 || 1) {
        // save jpeg
        std::string fname = "hevc_";
        fname += std::to_string(g_n);
        fname += ".jpeg";
        xffmpeg::ffmpeg_save_jpeg(frame, fname.c_str());
    }
    std::cout << "pic_type: " << xffmpeg::picture_type(frame->pict_type)
        << " key_frame: " << frame->key_frame
        << " pic_frame_num: " << frame->coded_picture_number
        //<< " frame_number: " << demuxer.GetCodecContext()->frame_number
        << " pts: " << frame->pts
        << " [" << frame->width << "," << frame->height
        << "\n";
}
#endif


//=======================================================================
// 
//=======================================================================

#if 0
FFMpeg_Rtsp_Decode2::FFMpeg_Rtsp_Decode2()
     : Super{}
{
}

FFMpeg_Rtsp_Decode2::~FFMpeg_Rtsp_Decode2()
{
}

int FFMpeg_Rtsp_Decode2::Loop()
{
    AVPacket* pkt = ::av_packet_alloc();
    int ret;
    xffmpeg::MicrosecondTimer timer;
    while (1) {
        // 此处如果是从rtsp server获取的流，这个耗时就是每帧耗时。
        timer.Start();
        // 3.2 从输出流读取一个packet
        ret = av_read_frame(ifmt_ctx_, pkt);
        timer.Stop();
        std::cout << "cost: " << timer.GetMilliseconds() << "\n";
        if (ret < 0) {
            break;
        }

        if (pkt->stream_index == stream_index_) {
            ParserDecode(pkt->data, pkt->size);
            //cb_(pkt->data, pkt->size, 0, {}, 0);
        }
        av_packet_unref(pkt);
    }
    return 0;
}

#endif

class RtspClient : public FFMpegClient
{
    using Super = FFMpegClient;
public:
    RtspClient(FrameCallback cb, const RtspParam* param)
        : Super(std::move(cb))
        , param_{*param}
    {
        Super::read_timeout_ = param_.read_timeout;
        Super::url_ = param_.url;
    }

    virtual ~RtspClient()
    {

    }

    virtual int Init() override
    {
        g_file.open("C:/github/estl/ffmpeg/ffmpeg_v4/rtsp.264");
        int ecode{};
        if (param_.protocol_type == TCP) {
            ecode = ::av_dict_set(&options_, "rtsp_transport", "tcp", 0);
            ecode = ::av_dict_set(&options_, "stimeout", "5000000", 0);
        } else if (param_.protocol_type == UDP) {
            ecode = ::av_dict_set(&options_, "rtsp_transport", "udp", 0);
            ecode = ::av_dict_set(&options_, "stimeout", "5000000", 0);
        }
        if (ecode < 0) {
            FFMPEGX_LOG_WARN("set rtsp_transport: %d failure", param_.protocol_type);
            return ecode;
        }
        return Super::Init();
    }

    RtspParam  param_;
};

class RtmpClient : public FFMpegClient
{
    using Super = FFMpegClient;
public:
    RtmpClient(FrameCallback cb, const RtmpParam* param)
        : Super(std::move(cb))
        , param_{*param}
    {
        Super::read_timeout_ = param_.read_timeout;
        Super::url_ = param_.url;
    }

    virtual ~RtmpClient()
    {

    }

    virtual int Init() override
    {
        g_file.open("C:/github/estl/ffmpeg/ffmpeg_v4/rtmp.264");
        int ecode{};
        ecode = ::av_dict_set(&options_, "stimeout", "5000000", 0);
        if (ecode < 0) {
            FFMPEGX_LOG_WARN("set stimeout: failure");
            return ecode;
        }
        assert(parser_.Init_H264() == 0);
        return Super::Init();
    }

#if 1
    virtual int OnReadPkt(AVPacket* pkt) override
    {
        int ecode{};
        int idx = pkt->stream_index;
        AVStream* in_stream = ifmt_ctx_->streams[idx];
        AVCodecParameters* in_codecpar = in_stream->codecpar;
        if (in_codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            g_file.write(pkt->data, pkt->size);
            std::ostringstream ostm{};
            AVPacket_to_string(pkt, ostm);
            FFMPEGX_LOG_INFO("stream_index: %d %d  %s", idx, in_codecpar->codec_id, ostm.str().c_str());


            parser_.AppendRawData(pkt->data, pkt->size);
            ecode = parser_.Parse();
            assert(ecode == 0);
            int ecode = parser_.PrepareDecode();
            assert(ecode == 0);
            AVPacket* pkt_h264 = parser_.GetPacket();
            if (pkt_h264->size >= 10) {
                FFMPEGX_LOG_INFO("stream_index: %d head: %d %d %d %d %d %d %d %d",
                    idx,
                    pkt_h264->data[0], pkt_h264->data[1], pkt_h264->data[2], pkt_h264->data[3],
                    pkt_h264->data[4], pkt_h264->data[5], pkt_h264->data[6], pkt_h264->data[7]
                );
            }
        }
        return 0;
    }
#endif

    MemParser parser_;
    RtmpParam  param_;
};

class FFMpegSdk
{
public:
    using RtspTask = std::function<void()>;

    FFMpegSdk();
    ~FFMpegSdk();

    int Init();
    void Cleanup();
    int StartPullRtsp(const RtspParam* param, FrameCallback user_cb, RtspHandle* hdl);
    int StopPullRtsp(RtspHandle hdl);
    void PostTask(RtspTask t);
    void Async_ClientReceivedFrame(RtspHandle hdl, std::shared_ptr<RtspRawFrame> frame);
    std::unique_ptr<RtspRawFrame> CreateRtspRawFrame();
private:
    void ThreadRun();
    void Async_CreateNewClient(RtspHandle hdl, std::shared_ptr<RtspParam> param, FrameCallback cb);
    void ClientStopPull(RtspHandle hdl);
    void OnCreateNewClient(RtspHandle new_hdl, std::shared_ptr<RtspParam> param, FrameCallback user_cb);
    void OnRawFrameReceived(RtspHandle hdl, std::shared_ptr<RtspRawFrame> frame);
    RtspClientImpl* FindClient(RtspHandle hdl);

private:
    std::atomic<bool>                   running_;
    std::atomic<RtspHandle>             next_hdl_;
    std::mutex                          mtx_;
    std::condition_variable             cond_;
    std::queue<RtspTask>                queue_;
    std::unordered_map<RtspHandle, std::shared_ptr<FFMpegClient>> clients_;
    std::thread                         thd_;
};

FFMpegSdk::FFMpegSdk()
    : running_{}
    , next_hdl_{}
    , mtx_{}
    , cond_{}
    , queue_{}
    , clients_{}
    , thd_{}
{
}

FFMpegSdk::~FFMpegSdk()
{
    Cleanup();
    if (thd_.joinable())
        thd_.join();
}

int FFMpegSdk::Init()
{
    running_ = true;
    std::thread temp_thd{
        [this] { this->ThreadRun(); }
    };
    std::swap(thd_, temp_thd);
    return 0;
}

void FFMpegSdk::Cleanup()
{
    running_ = false;
    PostTask([] {});
}

int FFMpegSdk::StartPullRtsp(const RtspParam* param, FrameCallback user_cb, RtspHandle* hdl)
{
    auto new_hdl = ++next_hdl_;
    auto p = std::make_shared<RtspParam>(*param);
    Async_CreateNewClient(new_hdl, std::move(p), std::move(user_cb));
    *hdl = new_hdl;
    return 0;
}

int FFMpegSdk::StopPullRtsp(RtspHandle hdl)
{
    PostTask([this, hdl]() mutable
        {
            this->ClientStopPull(hdl);
        });
    return 0;
}

void FFMpegSdk::Async_ClientReceivedFrame(RtspHandle hdl, std::shared_ptr<RtspRawFrame> frame)
{
    PostTask([this, hdl, frame_ex = std::move(frame)]() mutable
    {
        this->OnRawFrameReceived(hdl, std::move(frame_ex));
    });
}

std::unique_ptr<RtspRawFrame> FFMpegSdk::CreateRtspRawFrame()
{
    return std::make_unique<RtspRawFrame>();
}

void FFMpegSdk::PostTask(RtspTask t)
{
    std::lock_guard<std::mutex> lk{ mtx_ };
    queue_.emplace(std::move(t));
    cond_.notify_one();
}

void FFMpegSdk::ThreadRun()
{
    RtspTask task{};
    while (running_) {
        {
            std::unique_lock<std::mutex> lk{ mtx_ };
            cond_.wait(lk, [this]() { return !queue_.empty(); });
            task = std::move(queue_.front());
            queue_.pop();
        }
        try {
            task();
        }
        catch (...) {
        }
    }
}

void FFMpegSdk::Async_CreateNewClient(RtspHandle hdl, std::shared_ptr<RtspParam> param, FrameCallback cb)
{
    PostTask([this, hdl, param_ex = std::move(param), cb_ex = std::move(cb)]() mutable
    {
        this->OnCreateNewClient(hdl, std::move(param_ex), std::move(cb_ex));
    });
}

void FFMpegSdk::ClientStopPull(RtspHandle hdl)
{
    auto* client = FindClient(hdl);
    if (!client)
        return;
    client->Stop();
    clients_.erase(hdl);
}

void FFMpegSdk::OnCreateNewClient(RtspHandle new_hdl, std::shared_ptr<RtspParam> param, FrameCallback user_cb)
{
    std::shared_ptr<FFMpegClient> new_client{ CreateClient(user_cb, param.get() };
    int ecode = new_client->Init();
    if (ecode) {

    }
    clients_.emplace(new_hdl, std::move(new_client));
}

void FFMpegSdk::OnRawFrameReceived(RtspHandle hdl, std::shared_ptr<RtspRawFrame> frame)
{
    // 1. client不存在，移除此client
    // 2. client存在，执行user cb
    auto* client = FindClient(hdl);
    if (!client) {
        client->Stop();
        clients_.erase(hdl);
        client = nullptr;
        return;
    }
    client->user_cb_(client->hdl_, &frame->info_, frame->data_.data(), static_cast<std::int32_t>(frame->data_.size()));
}

FFMpegClient* FFMpegSdk::FindClient(RtspHandle hdl)
{
    auto it = clients_.find(hdl);
    if (it == clients_.end())
        return nullptr;
    return it->second.get();
}

FFMpegClient* CreateClient(FrameCallback cb, const RtspParam* param)
{
    return new RtspClient(cb, param);
}

FFMpegClient* CreateClient(FrameCallback cb, const RtmpParam* param)
{
    return new RtmpClient(cb, param);
}


} // namespace ffmpegx
