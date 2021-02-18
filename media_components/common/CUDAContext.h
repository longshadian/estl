#pragma once

#include <condition_variable>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include <cuda.h>
#include <cuda_runtime.h>

// GPU×ÊÔ´£¬ÏÔ´æ
class GpuResource
{
public:
    GpuResource()
        : gpu_buffer_()
        , sz_()
    {
    }

    ~GpuResource()
    {
        if (gpu_buffer_)
            cuMemFree(gpu_buffer_);
    }

    CUdeviceptr gpu_buffer_;

    std::size_t sz_;
};

class CUADContext
{
public:
    CUADContext()
        : cuContext{}
    {
    }

    ~CUADContext()
    {
    }

    int Init(int gpuid)
    {
        (void)gpuid;
        int ecode{};
        try {
            ecode = cuInit(0);
            if (ecode < 0) {
                logPrintWarn("cuInit error");
                return -1;
            }
                
            int nGpu = 0;
            ecode = cuDeviceGetCount(&nGpu);
            if (ecode < 0) {
                logPrintWarn("cuDeviceGetCount error");
                return -1;
            }
            if (gpuid < 0 || gpuid >= nGpu) {
                logPrintWarn("GPU ordinal out of range. Should be within [0, %d]", nGpu - 1);
                return -1;
            }

            ecode = createCudaContext(&cuContext, gpuid, 0);
            if (ecode < 0) {
                logPrintWarn("createCudaContext error");
                return -1;
            }
            return 0;
        } catch (const std::exception& e) {
            logPrintWarn("exception: %s", e.what());
            return -1;
        }
    }

    static int createCudaContext(CUcontext* cuContext, int iGpu, unsigned int flags)
    {
        int ecode{};
        CUdevice cuDevice = 0;
        ecode = cuDeviceGet(&cuDevice, iGpu);
        if (ecode < 0) {
            logPrintWarn("cuDeviceGet error");
            return -1;
        }
        char szDeviceName[80];
        ecode = cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice);
        if (ecode < 0) {
            logPrintWarn("cuDeviceGetName error");
            return -1;
        }
        logPrintInfo("GPU in use: %s",szDeviceName);
        ecode = cuCtxCreate(cuContext, flags, cuDevice);
        if (ecode < 0) {
            logPrintWarn("cuCtxCreate error");
            return -1;
        }
        return 0;
    }

    std::shared_ptr<GpuResource> Create(int width, int height)
    {
        CUdeviceptr gpu_mem{};
        cuCtxPushCurrent(cuContext);
        cuMemAlloc(&gpu_mem, width * height * 3);
        cudaMemset((uint8_t*)gpu_mem, 128, width * height * 3);
        cuCtxPopCurrent(NULL);

        auto p = std::make_shared<GpuResource>();
        p->sz_ = size_t(width * height * 3);
        std::swap(p->gpu_buffer_, gpu_mem);
        return p;
    }

    CUcontext cuContext;
};

