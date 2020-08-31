#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>

int ffmpeg_util_save_jpeg(AVFrame* frame, const char* out_name); 


#ifdef __cplusplus
}
#endif

