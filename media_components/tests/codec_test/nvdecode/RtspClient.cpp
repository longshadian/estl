#include "RtspClient.h"

#include "console_log.h"

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
    //logPrintInfo("--->save file: %d", demuxer.GetPacket()->size);
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
            if (g_n%30 != 0)
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
            if (g_n > 100 && true) {
                std::cout << "exit!!!" << std::endl;
                exit(0);
            }
        }
    } while (decode_result > 0);
    return 0;
}


RtspClient_Live555::RtspClient_Live555()
    : sdk_{std::make_unique<media::RtspSDK>()}
    , cb_{}
    , hdl_{}
{
}

RtspClient_Live555::~RtspClient_Live555()
{
}

int RtspClient_Live555::Init(std::string url, Callback_t cb)
{
    using namespace std::placeholders;

    cb_ = cb;
    sdk_->Init();
    media::RtspParam param{};
    param.protocol_type = media::EProtocolType::UDP;
    sdk_->StartPullRtsp(&param, url, std::bind(&RtspClient_Live555::FrameCB, this, _1, _2, _3, _4), &hdl_);

    parser_.Init_H265();
    num_ = 0;
    return 0;
}

int RtspClient_Live555::Loop()
{
    while (1) {
        std::this_thread::sleep_for(std::chrono::seconds{1});
    }
    return 0;
}

void RtspClient_Live555::FrameProcess(
    unsigned char* buffer,
    unsigned int buffer_length,
    unsigned numTruncatedBytes,
    struct timeval presentationTime,
    unsigned durationInMicroseconds
)
{
    cb_(buffer, buffer_length, numTruncatedBytes, presentationTime, durationInMicroseconds);
}

void RtspClient_Live555::FrameCB(media::RtspHandle hdl, const media::RtspRawFrameInfo* info, std::uint8_t* buffer, std::int32_t buffer_length)
{
    assert(hdl_ == hdl);
    if (info->codec_type != media::ECodecType::HEVC)
        return;
    logPrintInfo("info: %d [%dx%d] length: %d", info->codec_type, info->width, info->height, buffer_length);
    if (num_ == 0) {
        do {
            int ecode{};
            parser_.AppendRawData(buffer, buffer_length);
            ecode = parser_.Parse();
            if (ecode == 0)
                break;
            assert(ecode == 1);
            ecode = parser_.PrepareDecode();
            if (ecode < 0)
                break;
            ecode = parser_.PrepareCodecParameter();
            assert(ecode >= 0);

            std::ostringstream ostm{};
            xffmpeg::AVCodecParameters_to_string(parser_.GetCodecParameters(), ostm);
            std::cout << ostm.str() << "\n";
        } while (0);
    }

    cb_(buffer, buffer_length, 0, {}, 0);
}

//=======================================================================
// 
//=======================================================================

RtspClient_File_HEVC::RtspClient_File_HEVC()
    : cb_{}
    , fname_{}
{
}

RtspClient_File_HEVC::~RtspClient_File_HEVC()
{
}

int RtspClient_File_HEVC::Init(std::string url, Callback_t cb)
{
    cb_ = cb;
    fname_ = url;
    return 0;
}

int RtspClient_File_HEVC::Loop()
{
    FILE* f = std::fopen(fname_.c_str(), "rb");
    if (!f) {
        logPrintWarn("xxxx open file %s failure", fname_.c_str());
        return -1;
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
                xffmpeg::AVCodecContext_to_string(demuxer.GetCodecContext(), ostm);
                //std::cout << "----------->: " << ostm.str() << "\n";
                AVPacket& pkt = *demuxer.GetPacket();
                if (pkt.size >= 10) {
                    printf("head: %d %d %d %d %d %d %d %d\n", pkt.data[0], pkt.data[1], pkt.data[2], pkt.data[3],
                        pkt.data[4], pkt.data[5], pkt.data[6], pkt.data[7]
                    );
                }
            }
            ecode = demuxer.PrepareDecode();
            if (ecode < 0) {
                char str[128] = {0};
                ::av_strerror(ecode, str, sizeof(str));
                logPrintWarn("avcodec_send_pac  ket ecode: %d  reason: %s", ecode, str);
                continue;
            }

            //cb_(demuxer.GetPacket()->data, demuxer.GetPacket()->size, 0, {}, 0);
            ecode = DoDecodeHEVC(demuxer);
            if (ecode < 0) {
                logPrintWarn("DoDecodeHEVC failure");
                //return -1;
                continue;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{40});
        } while (parse_result > 0);
    }

    return 0;
}

//=======================================================================
// 
//=======================================================================

RtspClient_HEVC_Save::RtspClient_HEVC_Save()
    : cb_{}
    , fname_{}
    , ifmt_ctx{}
    , stream_mapping{}
    , stream_index{}
    , stream_mapping_size{}
    , duration{}
{
}

RtspClient_HEVC_Save::~RtspClient_HEVC_Save()
{
}

int RtspClient_HEVC_Save::Init(std::string url, Callback_t cb)
{
    std::string in_filename = url;
    cb_ = cb;

    int ret;
    AVDictionary* options{};
    if (1) {
        ::av_dict_set(&options, "rtsp_transport", "tcp", 0);
    }

    // 1. 打开输入
    // 1.1 读取文件头，获取封装格式相关信息
    if ((ret = avformat_open_input(&ifmt_ctx, in_filename.c_str(), 0, &options)) < 0) {
        printf("Could not open input file '%s'", in_filename.c_str());
        //goto end;
        return -1;
    }

    // 1.2 解码一段数据，获取流相关信息
    if ((ret = avformat_find_stream_info(ifmt_ctx, 0)) < 0) {
        printf("Failed to retrieve input stream information");
        //goto end;
        return -1;
    }

    av_dump_format(ifmt_ctx, 0, in_filename.c_str(), 0);

    // 2. 打开输出
    // 2.1 分配输出ctx
    stream_mapping_size = ifmt_ctx->nb_streams;
    stream_mapping = (int*)av_mallocz_array(stream_mapping_size, sizeof(*stream_mapping));
    if (!stream_mapping) {
        ret = AVERROR(ENOMEM);
        //goto end;
        return -1;
    }

    AVRational frame_rate;
    for (unsigned i = 0; i < ifmt_ctx->nb_streams; i++) {
        AVStream* in_stream = ifmt_ctx->streams[i];
        AVCodecParameters* in_codecpar = in_stream->codecpar;

        if (in_codecpar->codec_type != AVMEDIA_TYPE_AUDIO &&
            in_codecpar->codec_type != AVMEDIA_TYPE_VIDEO &&
            in_codecpar->codec_type != AVMEDIA_TYPE_SUBTITLE) {
            stream_mapping[i] = -1;
            continue;
        }

        if (in_codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            frame_rate = av_guess_frame_rate(ifmt_ctx, in_stream, NULL);
            AVRational r{};
            r.den = frame_rate.den;
            r.num = frame_rate.num;
            duration = (frame_rate.num && frame_rate.den ? av_q2d(r) : 0);
        }

        stream_mapping[i] = stream_index++;
    }
    return 0;
}

int RtspClient_HEVC_Save::Loop()
{
    AVPacket pkt;
    int ret;
    xffmpeg::MicrosecondTimer timer;
    while (1) {
        AVStream* in_stream;

        // 此处如果是从rtsp server获取的流，这个耗时就是每帧耗时。
        timer.Start();
        // 3.2 从输出流读取一个packet
        ret = av_read_frame(ifmt_ctx, &pkt);
        timer.Stop();
        std::cout << "cost: " << timer.GetMilliseconds();
        if (ret < 0) {
            break;
        }

        //std::ostringstream ostm{};
        //avpacket_to_string(&pkt, ostm);
        //std::cout << ostm.str() << "\n";

        in_stream = ifmt_ctx->streams[pkt.stream_index];
        if (pkt.stream_index >= stream_mapping_size ||
            stream_mapping[pkt.stream_index] < 0) {
            av_packet_unref(&pkt);
            continue;
        }

        int codec_type = in_stream->codecpar->codec_type;
        if (codec_type == AVMEDIA_TYPE_VIDEO) {
            int64_t d = (int64_t)(duration * AV_TIME_BASE);
            std::cout << d << "\n";
            //av_usleep((int64_t)(duration * AV_TIME_BASE));
            if (0) {
                if (pkt.size >= 4) {
                    printf("head: %d %d %d %d\n", pkt.data[0], pkt.data[1], pkt.data[2], pkt.data[3]);
                }
                //for (int i = 0; i != headn; ++i) {
                //}
            }
            pkt.stream_index = stream_mapping[pkt.stream_index];

            if (ret < 0) {
                printf("Error muxing packet\n");
                break;
            }
            FrameProcess(pkt.data, pkt.size, 0, {}, 0);
        }
        av_packet_unref(&pkt);
    }
    return 0;
}

int RtspClient_HEVC_Save::Fun()
{
    ::avformat_alloc_output_context2(&ofmt_ctx, nullptr, "HEVC", nullptr);
    if (!ofmt_ctx) {
        printf("Could not create output context\n");
        return -1;
    }

    out_stream = ::avformat_new_stream(ofmt_ctx, nullptr);
    if (!out_stream) {
        printf("Failed allocating output stream\n");
        return -1;
    }
    return 0;
}

void RtspClient_HEVC_Save::FrameProcess(
    unsigned char* buffer,
    unsigned int buffer_length,
    unsigned numTruncatedBytes,
    struct timeval presentationTime,
    unsigned durationInMicroseconds
)
{
    cb_(buffer, buffer_length, numTruncatedBytes, presentationTime, durationInMicroseconds);
}

