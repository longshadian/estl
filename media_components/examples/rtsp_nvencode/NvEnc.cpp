#include <iostream>
#include <sstream>

#include <RtspPoller.h>

#include "NvEnc.h"

#include "../common/console_log.h"
#include "../common/utility.h"

#include <cuda_runtime.h>

simplelogger::Logger* logger = simplelogger::LoggerFactory::CreateConsoleLogger();

/**
*   @brief  Utility function to create CUDA context
*   @param  cuContext - Pointer to CUcontext. Updated by this function.
*   @param  iGpu      - Device number to get handle for
*/
static void createCudaContext(CUcontext* cuContext, int iGpu, unsigned int flags)
{
    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, iGpu));
    char szDeviceName[80];
    ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    std::cout << "GPU in use: " << szDeviceName << std::endl;
    ck(cuCtxCreate(cuContext, flags, cuDevice));
}

static std::unique_ptr<NvEnc::FrameData> CreateFrameData(
    unsigned char* buffer,
    unsigned int buffer_length,
    unsigned numTruncatedBytes,
    struct timeval presentationTime,
    unsigned durationInMicroseconds,
    int64_t frame_num
)
{
    std::unique_ptr<NvEnc::FrameData> p = std::make_unique<NvEnc::FrameData>();
    p->frame_num_ = frame_num;
    p->numTruncatedBytes_ = numTruncatedBytes;
    p->presentationTime_ = presentationTime;
    p->durationInMicroseconds_ = durationInMicroseconds;
    p->data_.assign(buffer, buffer + buffer_length);
    return p;
}


template<class EncoderClass>
void InitializeEncoder(EncoderClass& pEnc, NvEncoderInitParam encodeCLIOptions, NV_ENC_BUFFER_FORMAT eFormat)
{
    NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
    NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };

    initializeParams.encodeConfig = &encodeConfig;
    pEnc->CreateDefaultEncoderParams(&initializeParams, encodeCLIOptions.GetEncodeGUID(), encodeCLIOptions.GetPresetGUID(), encodeCLIOptions.GetTuningInfo());
    encodeCLIOptions.SetInitParams(&initializeParams, eFormat);

    pEnc->CreateEncoder(&initializeParams);
}

NvEnc::NvEnc()
    : frame_num_()
    , cuContext()
    , rtsp_(std::make_unique<RtspPoller>())
    , nvdec_()
    , rtsp_thd_()
    , mtx_()
    , queue_()
    , cond_()
    , cropRect_()
    , resizeDim_()
    , mem_type_(MemoryType::Device)
    , img_buffer_()
    , out_f_()
{
}

NvEnc::~NvEnc()
{
    if (rtsp_thd_.joinable())
        rtsp_thd_.join();
    if (nvenc_) {
        nvenc_->DestroyEncoder();
    }
    if (out_f_)
        ::fclose(out_f_);
}

int NvEnc::Init(int gpu_num, cudaVideoCodec video_codec, MemoryType mem_type)
{
    mem_type_ = mem_type;
    try {
        ck(cuInit(0));
        int nGpu = 0;
        ck(cuDeviceGetCount(&nGpu));
        if (gpu_num < 0 || gpu_num >= nGpu) {
            std::cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
            return 1;
        }

        createCudaContext(&cuContext, gpu_num, 0);
        bool devie_mem = mem_type_ == MemoryType::Device;
        nvdec_ = std::make_unique<NvDecoder>(cuContext, devie_mem, video_codec, false, false, &cropRect_, &resizeDim_);
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}

int NvEnc::InitEncode(int nWidth, int nHeight, NV_ENC_BUFFER_FORMAT eFormat, NvEncoderInitParam encodeCLIOptions,
    std::string save_file)
{
    try {
        nvenc_ = std::make_unique<NvEncoderCuda>(cuContext, nWidth, nHeight, eFormat);
        InitializeEncoder(nvenc_, encodeCLIOptions, eFormat);
        out_f_ = ::fopen(save_file.c_str(), "wb");
        if (!out_f_)
            return -1;
        return 0;
    } catch (const std::exception& e) {
        CONSOLE_LOG_WARN << "exception: " << e.what();
        return -1;
    }
}

int NvEnc::StartPullRtspThread(std::string rtsp_uri)
{
    using namespace std::placeholders;

    RtspPollerParams params;
    params.url = std::move(rtsp_uri);
    params.frame_proc = std::bind(&NvEnc::FrameProc, this, _1, _2, _3, _4, _5);
    if (!rtsp_->Init(std::move(params))) {
        std::cout << "init error\n";
        return -1;
    }
    std::thread temp_thd([this]()
        {
            rtsp_->Loop();
        });
    std::swap(rtsp_thd_, temp_thd);
    return 0;
}

int NvEnc::StartDecoder(std::string pic_dir)
{
    std::unique_ptr<FrameData> p = nullptr;
    while (1) {
        {
            std::unique_lock<std::mutex> lk{mtx_};
            cond_.wait_for(lk, std::chrono::seconds(1), [this] { return !queue_.empty(); });
            while (queue_.empty())
                continue;
            p = std::move(queue_.front());
            queue_.pop_front();
        }
        if (p) {
            VideoDecode(*p, pic_dir);
            p = nullptr;
        }
    }
    return 0;
}

void NvEnc::FrameProc(
    unsigned char* buffer,
    unsigned int buffer_length,
    unsigned numTruncatedBytes,
    struct timeval presentationTime,
    unsigned durationInMicroseconds
)
{
    std::ostringstream ostm{};

    ostm << "FrameProc: " << ++frame_num_
        << " buffer_length: " << buffer_length
        << "\n";
    if (numTruncatedBytes > 0)
        ostm << " (with " << numTruncatedBytes << " bytes truncated)";
    char uSecsStr[6 + 1]; // used to output the 'microseconds' part of the presentation time
    sprintf(uSecsStr, "%06u", (unsigned)presentationTime.tv_usec);
    ostm << ".\tPresentation time: " << (int)presentationTime.tv_sec << "." << uSecsStr;
    ostm << "\n";
    //std::cout << ostm.str();

    {
        auto p = CreateFrameData(buffer, buffer_length, numTruncatedBytes, presentationTime, durationInMicroseconds, frame_num_);
        std::lock_guard<std::mutex> lk{mtx_};
        queue_.emplace_back(std::move(p));
        cond_.notify_all();
    }
}

int NvEnc::VideoDecode(FrameData& frame_data, const std::string& pic_dir)
{
    comm::MicrosecondTimer timer;
    timer.Start();

    int nFrameReturned = 0;
    uint8_t* pframe = nullptr;
    nFrameReturned = nvdec_->Decode(frame_data.data_.data(), frame_data.data_.size());
    /*
    bool bDecodeOutSemiPlanar = false;
    if (!nFrame && nFrameReturned)
        LOG(INFO) << dec.GetVideoInfo();
    bDecodeOutSemiPlanar = (nvdec_->GetOutputFormat() == cudaVideoSurfaceFormat_NV12) 
        || (nvdec_->GetOutputFormat() == cudaVideoSurfaceFormat_P016);
    */

    for (int i = 0; i < nFrameReturned; i++) {
        pframe = nvdec_->GetFrame();
        /*
        if (bOutPlanar && bDecodeOutSemiPlanar) {
            ConvertSemiplanarToPlanar(pFrame, dec.GetWidth(), dec.GetHeight(), dec.GetBitDepth());
        }
        fpOut.write(reinterpret_cast<char*>(pFrame), dec.GetFrameSize());
        */

        if (frame_data.frame_num_ > 0
            && frame_data.frame_num_ % 100 == 0) {
            comm::MicrosecondTimer mt1;
            mt1.Start();

            void* img_data = nullptr;
            size_t img_data_len = nvdec_->GetFrameSize();
            if (mem_type_ == MemoryType::Device) {
                img_buffer_.resize(nvdec_->GetFrameSize());
                cudaMemcpy(img_buffer_.data(), pframe, nvdec_->GetFrameSize(), cudaMemcpyDeviceToHost);
                img_data = img_buffer_.data();
            } else {
                img_data = pframe;
            }

            VideoEncode(img_data, img_data_len, frame_data.frame_num_);

            char buf[64] = {0};
            snprintf(buf, sizeof(buf), "%s/x_%d.yuv", pic_dir.c_str(), (int)frame_data.frame_num_);
            comm::SaveFile(buf, img_data, img_data_len);
            mt1.Stop();
            CONSOLE_LOG_INFO << "format: " << nvdec_->GetOutputFormat() 
                << " save file: " << buf << " cost: " << mt1.GetMilliseconds()
                << " width: " << nvdec_->GetWidth()
                << " height: " << nvdec_->GetHeight()
                << " bitdepth: " << nvdec_->GetBitDepth()
                << " size: " << nvdec_->GetFrameSize()
                ;
        }
    }

    timer.Stop();
    CONSOLE_LOG_INFO << "deocde cost: " << timer.GetMilliseconds() 
        << " nFrameReturned: " << nFrameReturned
        << " frame size: " << frame_data.data_.size() ;

    return 0;
}

int NvEnc::VideoEncode(void* pic_data, size_t len, int64_t frame_num)
{
    comm::MicrosecondTimer timer;
    timer.Start();
    std::vector<std::vector<uint8_t>> packet;
    if ((int)len == nvenc_->GetFrameSize()) {
        const NvEncInputFrame* input_frame = nvenc_->GetNextInputFrame();
        NvEncoderCuda::CopyToDeviceFrame(cuContext, pic_data, 0, (CUdeviceptr)input_frame->inputPtr,
            (int)input_frame->pitch,
            nvenc_->GetEncodeWidth(),
            nvenc_->GetEncodeHeight(),
            CU_MEMORYTYPE_HOST,
            input_frame->bufferFormat,
            input_frame->chromaOffsets,
            input_frame->numChromaPlanes
        );
        nvenc_->EncodeFrame(packet);
    } else {
        nvenc_->EndEncode(packet);
    }
    timer.Start();
    CONSOLE_LOG_INFO << "encode frame_num: " << frame_num << " data: " << len 
        << " pkg_size: " << packet.size() << " cost: " << timer.GetMilliseconds();

    for (const auto& vec : packet) {
        char buf[4] = {0,0,0,1};
        ::fwrite(buf, 1, 4, out_f_);
        ::fwrite(vec.data(), 1, vec.size(), out_f_);
    }
    return 0;
}


