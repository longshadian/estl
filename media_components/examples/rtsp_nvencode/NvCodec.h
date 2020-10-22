#pragma once

#if defined (_MSC_VER)

#elif defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wsign-compare"
    #pragma GCC diagnostic ignored "-Wunused-variable"
    #pragma GCC diagnostic ignored "-Wparentheses"
    #pragma GCC diagnostic ignored "-Wconversion"

    #include <NvCodec/NvDecoder/NvDecoder.h>
    #include <NvCodec/NvEncoder/NvEncoderCuda.h>
    #include <NvCodec/Utils/NvEncoderCLIOptions.h>

    #pragma GCC diagnostic pop
#else

#endif


