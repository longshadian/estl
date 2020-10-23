# 媒体组件库
rtsp拉流采用 [live555](http://live555.com/)

推流至nginx-rtmp服务器采用 [srs](https://github.com/ossrs/srs/wiki/v3_CN_Home)

编解码采用 [NVIDIA Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk)

SDK所需驱动版本如下：
```
sdk9: Video_Codec_SDK_9.1.23.zip
    * Windows: Driver version 436.15  or higher
    * Linux:   Driver version 435.21  or higher
    * CUDA 10.0 Toolkit 

sdk10: Video_Codec_SDK_10.0.26.zip
    * Windows: Driver version 445.87 or higher
    * Linux: Driver version 450.51 or higher
    * CUDA 10.1 or higher Toolkit
    * CUDA 11.0 or higher is needed if GA100 GPU is used
```

## 内容说明 
```
examples: 拉流，推流，NvCodec编解码例子。
src: 分装后的.a和.so
tests: 算法测试内容
```

