#pragma once

#if defined (_MSC_VER)

#elif defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wsign-compare"
    #pragma GCC diagnostic ignored "-Wunused-variable"
    #pragma GCC diagnostic ignored "-Wparentheses"
    #pragma GCC diagnostic ignored "-Wconversion"

    #include <NvCodec10/NvDecoder/NvDecoder.h>
    #include <NvCodec10/NvEncoder/NvEncoderCuda.h>
    #include <NvCodec10/Utils/Logger.h>
    #include <NvCodec10/Utils/NvCodecUtils.h>
    #include <NvCodec10/Utils/NvEncoderCLIOptions.h>

    #pragma GCC diagnostic pop
#else

#endif

