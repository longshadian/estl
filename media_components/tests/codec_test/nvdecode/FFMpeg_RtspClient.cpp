#include "FFMpeg_RtspClient.h"

#include "file_utils.h"

static std::string g_save_file_path
= "/home/bolan/works/vsremote/media_components/out/build/Linux-GCC-Debug-234/bin/1.265";

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
    //logPrintInfo("--->save file: %d", demuxer.getPacket()->size);
    SaveVideo(g_save_file_path, demuxer.GetPacket()->data, demuxer.GetPacket()->size);

    ostm.str("");
    xffmpeg::AVPacket_to_string(demuxer.GetPacket(), ostm);
    //std::cout << "avpacket: " << ostm.str() << "\n";

    ostm.str("");
    xffmpeg::AVCodecContext_to_string(demuxer.GetCodecContext(), ostm);
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
            if (g_n % 30 != 0)
                continue;
#if 1
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
                << " frame_number: " << demuxer.GetCodecContext()->frame_number
                << " pts: " << frame->pts
                << " [" << frame->width << "," << frame->height
                << "\n";
#endif
            if (g_n > 100 && 0) {
                std::cout << "exit!!!" << std::endl;
                exit(0);
            }
        }
    } while (decode_result > 0);
    return 0;
}




//=======================================================================
// 
//=======================================================================

FFMpeg_Base::FFMpeg_Base()
    : cb_{}
    , url_{}
{
}

FFMpeg_Base::~FFMpeg_Base()
{
}

int FFMpeg_Base::Init(std::string url, Callback_t cb)
{
    cb_ = cb;
    url_ = url;
    return 0;
}

int FFMpeg_Base::Loop()
{
    return 0;
}


FFMpeg_File::FFMpeg_File()
    : Super{}
    , demuxer_{}
{
}

FFMpeg_File::~FFMpeg_File()
{
}

int FFMpeg_File::Init(std::string url, Callback_t cb)
{
    if (Super::Init(url, cb) < 0) {
        return -1;
    }
    demuxer_.Init_H265();
    return 0;
}

int FFMpeg_File::Loop()
{
    FILE* f = std::fopen(url_.c_str(), "rb");
    if (!f) {
        logPrintWarn("xxxx open file %s failure", url_.c_str());
        return -1;
    }
    std::vector<char> buf(1024 * 1000, '\0');
    while (1) {
        int n = fread(buf.data(), 1, buf.size(), f);
        if (n == 0) {
            logPrintWarn("fread file n == 0");
            break;
        }
        auto* pbuf = buf.data();
        int len = (int)buf.size();
        ParserDecode(pbuf, len);
    }
    return 0;
}

int FFMpeg_File::ParserDecode(const void* pbuf, size_t len)
{
    int parse_result = 0;
    int ecode{};
    demuxer_.AppendRawData(pbuf, len);
    xffmpeg::MicrosecondTimer tm{};
    do {
        tm.Start();
        parse_result = demuxer_.Parse();
        tm.Stop();
        if (parse_result < 0) {
            logPrintWarn("parse pkg failure");
            return -1;
        }
        //if (parse_result > 0) {
        logPrintInfo("parse pkg result: %d cost: %f", parse_result, tm.GetMilliseconds());
        //}

        if (parse_result == 0) break;

        if (1) {
            cb_(demuxer_.GetPacket()->data, demuxer_.GetPacket()->size, 0, {}, 0);
            std::this_thread::sleep_for(std::chrono::milliseconds{40});
            continue;
        }
        {
            std::ostringstream ostm{};
            xffmpeg::AVCodecContext_to_string(demuxer_.GetCodecContext(), ostm);
            //std::cout << "----------->: " << ostm.str() << "\n";
            AVPacket& pkt = *demuxer_.GetPacket();
            if (pkt.size >= 10) {
                printf("head: %d %d %d %d %d %d %d %d\n", pkt.data[0], pkt.data[1], pkt.data[2], pkt.data[3],
                    pkt.data[4], pkt.data[5], pkt.data[6], pkt.data[7]
                );
            }
        }
        ecode = demuxer_.PrepareDecode();
        if (ecode < 0) {
            char str[128] = { 0 };
            ::av_strerror(ecode, str, sizeof(str));
            logPrintWarn("avcodec_send_pac  ket ecode: %d  reason: %s", ecode, str);
            continue;
        }

        //cb_(demuxer_.pkt_->data, demuxer_.pkt_->size, 0, {}, 0);
        ecode = DoDecodeHEVC(demuxer_);
        if (ecode < 0) {
            logPrintWarn("DoDecodeHEVC failure");
            //return -1;
            continue;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{ 40 });
    } while (parse_result > 0);
    return 0;
}

//=======================================================================
// 
//=======================================================================

FFMpeg_Rtsp::FFMpeg_Rtsp()
    : ifmt_ctx_{}
    , stream_index_{}
    , vid_stream_{}
{
}

FFMpeg_Rtsp::~FFMpeg_Rtsp()
{
    if (ifmt_ctx_)
        ::avformat_close_input(&ifmt_ctx_);
}

int FFMpeg_Rtsp::Init(std::string url, Callback_t cb)
{
    if (Super::Init(url, cb) < 0)
        return -1;

    int ret;
    AVDictionary* options{};
    if (1) {
        ::av_dict_set(&options, "rtsp_transport", "tcp", 0);
    }

    // 1. 打开输入
    // 1.1 读取文件头，获取封装格式相关信息
    if ((ret = avformat_open_input(&ifmt_ctx_, url_.c_str(), 0, &options)) < 0) {
        printf("Could not open input file '%s'", url_.c_str());
        //goto end;
        return -1;
    }

    // 1.2 解码一段数据，获取流相关信息
    if ((ret = avformat_find_stream_info(ifmt_ctx_, 0)) < 0) {
        printf("Failed to retrieve input stream information");
        //goto end;
        return -1;
    }

    av_dump_format(ifmt_ctx_, 0, url_.c_str(), 0);
    for (unsigned i = 0; i < ifmt_ctx_->nb_streams; i++) {
        AVStream* in_stream = ifmt_ctx_->streams[i];
        AVCodecParameters* in_codecpar = in_stream->codecpar;
        if (in_codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            vid_stream_ = in_stream;
            stream_index_ = i;
            break;
        }
    }
    return 0;
}

int FFMpeg_Rtsp::Loop()
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

        std::cout << "stream index: " << pkt->stream_index << " " << stream_index_ << "\n";
        if (pkt->stream_index == stream_index_) {
            OnReadPkt(pkt);
        }
        av_packet_unref(pkt);
    }
    return 0;
}

//=======================================================================
// 
//=======================================================================

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


//=======================================================================
// 
//=======================================================================

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

