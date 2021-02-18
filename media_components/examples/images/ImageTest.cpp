#include <iostream>
#include <sstream>

#include "common/utility.h"
#include "common/console_log.h"
#include "common/CUDAContext.h"

namespace sdk9_img_test
{

static int ResizeImage(CUcontext cuContext,
    unsigned char* vdata, int vdata_width, int vdata_height,   // 原始图像数据
    std::shared_ptr<GpuResource>& pbuffer, int dst_width, int dst_height,    // 临时buffer
    std::shared_ptr<GpuResource>& dst                               // 最终输出
)
{
    ///假如图像长和宽 不是1920 和 1080 的情况下 就进行缩放  存入刚申请的空间
    if (!pbuffer) {
        pbuffer = std::make_shared<GpuResource>();
        cuCtxPushCurrent(cuContext);
        cuMemAlloc(&pbuffer->gpu_buffer_, dst_width * dst_height * 3 / 2);
        cuCtxPopCurrent(NULL);
    }
    if (!pbuffer) {
        return -1;
    }
    cuCtxPushCurrent(cuContext);
    ResizeNv12((uint8_t*)pbuffer->gpu_buffer_, dst_width, dst_width, dst_height,
        vdata, vdata_width, vdata_width, vdata_height);
    cuCtxPopCurrent(NULL);

    // 复制缩放后的数据至 gpu_res
    cuCtxPushCurrent(cuContext);
    ///YUV转BGR
    Nv12ToBgr((uint8_t*)pbuffer->gpu_buffer_, (int)dst_width, (uint8_t*)dst->gpu_buffer_,
        dst_width, dst_width, dst_height);
    cuCtxPopCurrent(NULL);
    return 0;
}

bool TestImg()
{

    return true;
}

} // namespace sdk9_img_test

