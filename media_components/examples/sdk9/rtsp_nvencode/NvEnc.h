#pragma once

#include <condition_variable>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include <RtspPoller.h>

#include "../NvCodec9.h"

class NvEnc
{
public:
    enum class MemoryType
    {
        Device = 0,
        Host = 1,
    };

    struct FrameData
    {
        int64_t frame_num_{};
        std::vector<uint8_t> data_{};
        unsigned numTruncatedBytes_{};
        struct timeval presentationTime_{};
        unsigned durationInMicroseconds_{};
    };

public:
    NvEnc();
    ~NvEnc();

    void FrameProc(
        unsigned char* buffer,
        unsigned int buffer_length,
        unsigned numTruncatedBytes,
        struct timeval presentationTime,
        unsigned durationInMicroseconds
    );

    int Init(int gpu_num, cudaVideoCodec video_codec, MemoryType mem_type = MemoryType::Device);
    int InitEncode(int nWidth, int nHeight, NV_ENC_BUFFER_FORMAT eFormat, NvEncoderInitParam encodeCLIOptions,
         std::string save_file);
    int StartPullRtspThread(std::string rtsp_uri);
    int StartDecoder(std::string pic_dir);

private:
    int VideoDecode(FrameData& frame_data, const std::string& pic_dir);
    int VideoEncode(void* pic_data, size_t len, int64_t frame_num);

    int frame_num_;
    CUcontext cuContext;
    std::unique_ptr<RtspPoller> rtsp_;
    std::unique_ptr<NvDecoder> nvdec_;
    std::thread rtsp_thd_;

    std::mutex mtx_;
    std::list<std::unique_ptr<FrameData>> queue_;
    std::condition_variable cond_;
    Rect cropRect_;
    Dim resizeDim_;
    MemoryType mem_type_;
    std::vector<char> img_buffer_;

    std::unique_ptr<NvEncoderCuda> nvenc_;
    FILE* out_f_;
}; 


