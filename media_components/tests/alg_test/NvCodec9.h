#pragma once

#if defined (_MSC_VER)

#elif defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wsign-compare"
    #pragma GCC diagnostic ignored "-Wunused-variable"
    #pragma GCC diagnostic ignored "-Wparentheses"
    #pragma GCC diagnostic ignored "-Wconversion"

    #include <NvCodec9/NvDecoder/NvDecoder.h>
    #include <NvCodec9/NvEncoder/NvEncoderCuda.h>
    #include <NvCodec9/Utils/Logger.h>
    #include <NvCodec9/Utils/NvCodecUtils.h>
    #include <NvCodec9/Utils/NvEncoderCLIOptions.h>

    #pragma GCC diagnostic pop
#else

#endif

class NvCodec9_Helper
{
public:
static bool Decode(NvDecoder* handler, const uint8_t* pData, int nSize, std::vector<uint8_t*>* frames,
        uint32_t flags = 0, int64_t** ppTimestamp = NULL, int64_t timestamp = 0, CUstream stream = 0)
    {
        uint8_t** ppframe = nullptr;
        int pnFrameReturned = 0;
        bool succ = handler->Decode(pData, nSize, &ppframe, &pnFrameReturned, flags,  ppTimestamp, timestamp, stream);

        for (int i = 0; i != pnFrameReturned; ++i) {
            frames->push_back(ppframe[i]);
        }
        return succ;
    }

};



