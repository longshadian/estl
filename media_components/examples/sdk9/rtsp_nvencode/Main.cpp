#include <iostream>
#include <sstream>

#include "NvEnc.h"

void ShowEncoderCapability()
{
    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    std::cout << "Encoder Capability" << std::endl << std::endl;
    for (int iGpu = 0; iGpu < nGpu; iGpu++) {
        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, iGpu));
        char szDeviceName[80];
        ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
        CUcontext cuContext = NULL;
        ck(cuCtxCreate(&cuContext, 0, cuDevice));
        NvEncoderCuda enc(cuContext, 1920, 1080, NV_ENC_BUFFER_FORMAT_NV12);

        std::cout << "GPU " << iGpu << " - " << szDeviceName << std::endl << std::endl;
        std::cout << "\tH264:\t\t" << "  " <<
            (enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID,
                NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES) ? "yes" : "no") << std::endl <<
            "\tH264_444:\t" << "  " <<
            (enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID,
                NV_ENC_CAPS_SUPPORT_YUV444_ENCODE) ? "yes" : "no") << std::endl <<
            "\tH264_ME:\t" << "  " <<
            (enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID,
                NV_ENC_CAPS_SUPPORT_MEONLY_MODE) ? "yes" : "no") << std::endl <<
            "\tH264_WxH:\t" << "  " <<
            (enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID,
                NV_ENC_CAPS_WIDTH_MAX)) << "*" <<
            (enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_HEIGHT_MAX)) << std::endl <<
            "\tHEVC:\t\t" << "  " <<
            (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID,
                NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES) ? "yes" : "no") << std::endl <<
            "\tHEVC_Main10:\t" << "  " <<
            (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID,
                NV_ENC_CAPS_SUPPORT_10BIT_ENCODE) ? "yes" : "no") << std::endl <<
            "\tHEVC_Lossless:\t" << "  " <<
            (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID,
                NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE) ? "yes" : "no") << std::endl <<
            "\tHEVC_SAO:\t" << "  " <<
            (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID,
                NV_ENC_CAPS_SUPPORT_SAO) ? "yes" : "no") << std::endl <<
            "\tHEVC_444:\t" << "  " <<
            (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID,
                NV_ENC_CAPS_SUPPORT_YUV444_ENCODE) ? "yes" : "no") << std::endl <<
            "\tHEVC_ME:\t" << "  " <<
            (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID,
                NV_ENC_CAPS_SUPPORT_MEONLY_MODE) ? "yes" : "no") << std::endl <<
            "\tHEVC_WxH:\t" << "  " <<
            (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID,
                NV_ENC_CAPS_WIDTH_MAX)) << "*" <<
            (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_HEIGHT_MAX)) << std::endl;

        std::cout << std::endl;

        enc.DestroyEncoder();
        ck(cuCtxDestroy(cuContext));
    }
}

int main()
{
    if (0) {
        ShowEncoderCapability();
        return 0;
    }

    // 拉取rtsp流数据，解码生成图片，
    // 把图片编码，生成视频流，保存到本地
    //std::string url = "rtsp://192.168.1.95:8554/yintai.264";
    std::string url = "rtsp://192.168.1.95:8554/beijing.264";
    std::string pic_dir = "/home/bolan/works/pic";
    std::string out_video_file = "/home/bolan/works/pic/beijin.264";

    NvEnc dec;
    int ecode = dec.Init(0, cudaVideoCodec_H264, NvEnc::MemoryType::Host);
    if (ecode)
        return -1;
    ecode = dec.InitEncode(1920, 1080, NV_ENC_BUFFER_FORMAT_NV12, NvEncoderInitParam(), out_video_file);
    if (ecode)
        return -1;
    ecode = dec.StartPullRtspThread(url);
    if (ecode)
        return -1;
    dec.StartDecoder(pic_dir);

    return 0;
}

