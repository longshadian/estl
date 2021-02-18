#pragma once

#if defined(_MSC_VER)

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
