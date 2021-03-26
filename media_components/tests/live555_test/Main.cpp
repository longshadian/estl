#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>
#include <functional>

#include <RtspSDK.h>


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

    void FrameProc2(
        media::RtspHandle hdl, const media::RtspRawFrameInfo* info,
        std::uint8_t* buffer, std::int32_t buffer_length
    );

    int Start(std::string rtsp_uri, std::string file_name);

    FILE* f_;
    int frame_num_;
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

    media::RtspSDK impl;
    impl.Init();

    media::RtspParam param{};
    param.protocol_type = media::EProtocolType::UDP;
    media::RtspHandle hdl{};
    auto f = std::bind(&SaveFile::FrameProc2, this, _1, _2, _3, _4);
    impl.StartPullRtsp(&param, rtsp_uri, f, &hdl);
    while (1) {
        std::this_thread::sleep_for(std::chrono::seconds{1});
    }
    return 0;
}

void SaveFile::FrameProc2(
    media::RtspHandle hdl, const media::RtspRawFrameInfo* info,
    std::uint8_t* buffer, std::int32_t buffer_length
)
{
#if 1
    std::ostringstream ostm{};

    ostm << "FrameProc: " << ++frame_num_
        << " buffer_length: " << buffer_length
        << "\n";
    ostm << "\nNAL:( ";
    for (int i = 0; i != 6; ++i) {
        ostm << int(buffer[i]) << " ";
    }
    ostm << " )";
    std::cout << ostm.str() << "\n";
#endif

    ::fwrite(buffer, 1, buffer_length, f_);
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
        << "\n";
    if (numTruncatedBytes > 0)
        ostm << " (with " << numTruncatedBytes << " bytes truncated)";
    char uSecsStr[6 + 1]; // used to output the 'microseconds' part of the presentation time
    sprintf(uSecsStr, "%06u", (unsigned)presentationTime.tv_usec);
    ostm << ".\tPresentation time: " << (int)presentationTime.tv_sec << "." << uSecsStr;

    ostm << "\nNAL:( ";
    for (int i = 0; i != 6; ++i) {
        ostm << int(buffer[i]) << " ";
    }
    ostm << " )";
    std::cout << ostm.str() << "\n";
#endif

    ::fwrite(buffer, 1, buffer_length, f_);
}

int main()
{
    // 拉取rtsp流数据，保存到本地

#if 1
    //std::string url = "rtsp://192.168.1.95:8554/yf.264";
    std::string url = "rtsp://admin:s6Gw%25m%24X%24viqT@192.168.70.18:554";
    std::string file_name = "./yf.264";
#else 
    std::string url = "rtsp://192.168.16.231/airport2.265";
    std::string file_name = "./yf.265";
#endif

    SaveFile sf;
    if (sf.Start(url, file_name) != 0) {
        return -1;
    }
    return 0;
}

