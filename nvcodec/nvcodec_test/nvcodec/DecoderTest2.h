#pragma once

#include <memory>

class NvDecoder;

class CUDA_DecoderTest2
{
public:
    CUDA_DecoderTest2();
    ~CUDA_DecoderTest2();

    int Init();

private:
    std::unique_ptr<NvDecoder> nv_decoder_;
};


