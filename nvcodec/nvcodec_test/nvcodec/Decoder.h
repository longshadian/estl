#pragma once

#include <memory>

class CUDA_Decoder
{
public:
    CUDA_Decoder();
    ~CUDA_Decoder();

    int Init();


private:
    //std::unique_ptr<NvDecoder> nv_decoder_;
};


