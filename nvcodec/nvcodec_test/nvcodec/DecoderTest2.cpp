#include <string>
#include <iostream>
#include <fstream>

#include "DecoderTest2.h"
#include "MyFFmpegDemuxer.h"

#include <cuda.h>
#include "NvDecoder/NvDecoder.h"
#include "Utils/NvCodecUtils.h"
#include "Utils/FFmpegDemuxer.h"
//#include "AppDecUtils.h"



static bool SaveFile(const std::string& fname, const void* data, size_t len);
static bool SaveFile(const char* fname, const void* data, size_t len);
static void createCudaContext(CUcontext* cuContext, int iGpu, unsigned int flags);
static void DecodeMediaFile(CUcontext cuContext, const char* szInFilePath, const char* szOutFilePath, bool bOutPlanar,
    const Rect& cropRect, const Dim& resizeDim);


void ConvertSemiplanarToPlanar(uint8_t *pHostFrame, int nWidth, int nHeight, int nBitDepth);

/**
*   @brief  Utility function to create CUDA context
*   @param  cuContext - Pointer to CUcontext. Updated by this function.
*   @param  iGpu      - Device number to get handle for
*/
void createCudaContext(CUcontext* cuContext, int iGpu, unsigned int flags)
{
    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, iGpu));
    char szDeviceName[80];
    ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    std::cout << "GPU in use: " << szDeviceName << std::endl;
    ck(cuCtxCreate(cuContext, flags, cuDevice));
}

bool SaveFile(const std::string& fname, const void* data, size_t len)
{
    return SaveFile(fname.c_str(), data, len);
}

bool SaveFile(const char* fname, const void* data, size_t len)
{
    std::ofstream ofstm(fname, std::ios::out | std::ios::binary);
    if (!ofstm) {
        return false;
    }
     ofstm.write(reinterpret_cast<const char*>(data), len);
     ofstm.close();
     return true;
}

CUDA_DecoderTest2::CUDA_DecoderTest2()
{

}

CUDA_DecoderTest2::~CUDA_DecoderTest2()
{
}

int CUDA_DecoderTest2::Init()
{
    const char* szInFilePath = "/home/bolan/works/123.264";
    const char* szOutFilePath = "/home/bolan/works/pic2/123";
    bool bOutPlanar = false;
    int iGpu = 0;
    Rect cropRect = {};
    Dim resizeDim = {};

    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    if (iGpu < 0 || iGpu >= nGpu) {
        std::cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
        return 1;
    }

    CUcontext cuContext = NULL;
    createCudaContext(&cuContext, iGpu, 0);

    std::cout << "Decode with demuxing." << std::endl;
    DecodeMediaFile(cuContext, szInFilePath, szOutFilePath, bOutPlanar, cropRect, resizeDim);
}
   
/**
*   @brief  Function to decode media file and write raw frames into an output file.
*   @param  cuContext     - Handle to CUDA context
*   @param  szInFilePath  - Path to file to be decoded
*   @param  szOutFilePath - Path to output file into which raw frames are stored
*   @param  bOutPlanar    - Flag to indicate whether output needs to be converted to planar format
*   @param  cropRect      - Cropping rectangle coordinates
*   @param  resizeDim     - Resizing dimensions for output
*/
void DecodeMediaFile(CUcontext cuContext, const char* szInFilePath, const char* szOutFilePath, bool bOutPlanar,
    const Rect& cropRect, const Dim& resizeDim)
{
    MyFFmpegDemuxer demuxer{};
    int ret = demuxer.Init();
    if (ret < 0) {
        LOG(WARNING) << "demuxer init failure.";
        return;
    }

    NvDecoder dec(cuContext, false, cudaVideoCodec_H264, false, false, &cropRect, &resizeDim);

    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
    uint8_t* pVideo = NULL, * pFrame;
    bool bDecodeOutSemiPlanar = false;

    std::vector<uint8_t> buf(1024 * 1024, '\0');
    FILE* f = std::fopen(szInFilePath, "rb");
    if (!f) {
        LOG(WARNING) << dec.GetVideoInfo() << " open file " << szInFilePath << " failure!";
        return;
    }

    int pic_num = 0;
    int readn = 0;
    int n = 0;
    do {
        do {
            ret = demuxer.Demux(&pVideo, &nVideoBytes);
            if (ret < 0) {
                LOG(WARNING) << " demuxer demux failuer";
                return;
            } else if (ret == 0) {
                readn = std::fread(buf.data(), 1, buf.size(), f);
                if (readn == 0) {
                    LOG(INFO) <<" read file " << szInFilePath << " EOF!";
                    break;
                }
                demuxer.data_ = buf.data();
                demuxer.data_size_ = readn;
            }
        } while (ret == 0);
        ++n;
        if (n < 100)
            continue;
        if (n % 80 == 0) {
            LOG(WARNING) << "discard pkg n: " << n << " " << pic_num;
            continue;
        }

        //pVideo = buf.data();
        //nVideoBytes = (int)readn;

        nFrameReturned = dec.Decode(pVideo, nVideoBytes);
        if (!nFrame && nFrameReturned)
            LOG(INFO) << dec.GetVideoInfo();

        bDecodeOutSemiPlanar = (dec.GetOutputFormat() == cudaVideoSurfaceFormat_NV12) || (dec.GetOutputFormat() == cudaVideoSurfaceFormat_P016);

        for (int i = 0; i < nFrameReturned; i++) {
            pFrame = dec.GetFrame();
            if (bOutPlanar && bDecodeOutSemiPlanar) {
                ConvertSemiplanarToPlanar(pFrame, dec.GetWidth(), dec.GetHeight(), dec.GetBitDepth());
            }
            ++pic_num;

            std::ostringstream ostm;
            ostm << szOutFilePath <<  "-"  << pic_num << ".yuv";
            std::string fname = ostm.str();

            if (pic_num % 1000 == 0) {
                int ret = SaveFile(fname, pFrame, dec.GetFrameSize());
            }
            break;
        }
        nFrame += nFrameReturned;
        /*
        if (pic_num >= 100 * 20)
            break;
            */
    } while (1);

    std::vector <std::string> aszDecodeOutFormat = { "NV12", "P016", "YUV444", "YUV444P16" };
    if (bOutPlanar) {
        aszDecodeOutFormat[0] = "iyuv";   aszDecodeOutFormat[1] = "yuv420p16";
    }
    std::cout << "Total frame decoded: " << nFrame << std::endl
        << "Saved in file " << szOutFilePath << " in "
        << aszDecodeOutFormat[dec.GetOutputFormat()]
        << " format" << std::endl;
}

void TestH264_DecoderTest2()
{
    CUDA_DecoderTest2 dec;
    dec.Init();
}

