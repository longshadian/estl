#include <iostream>
#include <sstream>

#include <RtspPoller.h>
#include <xffmpeg/xffmpeg.h>

#include "console_log.h"

class SaveFile
{
public:
    SaveFile();
    ~SaveFile();

    void FrameProc(
        unsigned char* buffer,
        unsigned int buffer_length,
        unsigned numTruncatedBytes,
        struct timeval presentationTime,
        unsigned durationInMicroseconds
    );

    int Start(std::string rtsp_uri, std::string file_name);

    FILE* f_;
    int frame_num_;

    xffmpeg::MemParser mem_parser_;
};

SaveFile::SaveFile()
    : f_(nullptr)
    , frame_num_()
{
}

SaveFile::~SaveFile()
{
    if (f_)
        ::fclose(f_);
}

int SaveFile::Start(std::string rtsp_uri, std::string file_name)
{
    using namespace std::placeholders;

    f_ = ::fopen(file_name.c_str(), "wb");
    if (!f_) {
        std::cout << "open file error: " << file_name << "\n";
        return -1;
    }

    RtspPoller impl;
    RtspPollerParams params;
    params.url = std::move(rtsp_uri);
    params.frame_proc = std::bind(&SaveFile::FrameProc, this, _1, _2, _3, _4, _5);
    if (!impl.Init(std::move(params))) {
        std::cout << "init error\n";
        return -1;
    }
    impl.Loop();
    return 0;
}

void SaveFile::FrameProc(
    unsigned char* buffer,
    unsigned int buffer_length,
    unsigned numTruncatedBytes,
    struct timeval presentationTime,
    unsigned durationInMicroseconds
)
{
#if 1
    std::ostringstream ostm{};

    ostm << "FrameProc: " << ++frame_num_
        << " buffer_length: " << buffer_length
        << " ";
    if (numTruncatedBytes > 0)
        ostm << " (with " << numTruncatedBytes << " bytes truncated)";
    char uSecsStr[6 + 1]; // used to output the 'microseconds' part of the presentation time
    sprintf(uSecsStr, "%06u", (unsigned)presentationTime.tv_usec);
    ostm << ".\tPresentation time: " << (int)presentationTime.tv_sec << "." << uSecsStr;

    ostm << " NAL:( ";
    for (int i = 0; i != 6; ++i) {
        ostm << int(buffer[i]) << " ";
    }
    ostm << " )";
    std::cout << ostm.str() << "\n";
#endif

    static int idx = 0;
    if (1) {
#if 0
        if (idx == 0 || 1) {
            mem_parser_.AppendRawData(buffer, buffer_length);
        } else {
            mem_parser_.AppendRawData(buffer + 4, buffer_length - 4);
        }
#else
        mem_parser_.AppendRawData(buffer + 4, buffer_length - 4);
#endif
        ++idx;
        //mem_parser_.AppendRawData(buffer + 4, buffer_length - 4);
        //mem_parser_.AppendRawData(buffer, buffer_length);
        int parse_result{};
        int ecode{};
        do {
            parse_result = mem_parser_.Parse();
            if (parse_result < 0) {
                logPrintWarn("parser error ecode: %d", parse_result);
                return;
            }
            if (parse_result == 0) {
                //logPrintInfo("parser no packet ecode == 0");
                return;
            }
            ecode = mem_parser_.PrepareDecode();
            if (ecode < 0) {
                logPrintWarn("parser prepare decode ecode: %d", ecode);
                continue;
            }
            logPrintInfo("--------> write: ");
            //::fwrite(buffer, 1, buffer_length, f_);
            ::fwrite(mem_parser_.pkt_->data, 1, mem_parser_.pkt_->size, f_);
        } while (parse_result > 0);
    }
}

int TestRtsp_Live555()
{
    // 拉取rtsp流数据，保存到本地
    SaveFile sf;
#if 0
    std::string url = "rtsp://192.168.1.95:8554/beijing.264";
    std::string file_name = "./beijing.264";
    sf.mem_parser_.Init_H264();
#else 
    std::string url = "rtsp://192.168.16.231/airport2.265";
    std::string file_name = "./airport2.265";
    sf.mem_parser_.Init_H265();
#endif

    if (sf.Start(url, file_name) != 0) {
        return -1;
    }
    return 0;
}

