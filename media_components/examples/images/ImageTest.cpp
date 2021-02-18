#include <iostream>
#include <sstream>

#include "common/utility.h"
#include "common/console_log.h"
#include "common/CUDAContext.h"

namespace sdk9_img_test
{

static int ResizeImage(CUcontext cuContext,
    unsigned char* vdata, int vdata_width, int vdata_height,   // ԭʼͼ������
    std::shared_ptr<GpuResource>& pbuffer, int dst_width, int dst_height,    // ��ʱbuffer
    std::shared_ptr<GpuResource>& dst                               // �������
)
{
    ///����ͼ�񳤺Ϳ� ����1920 �� 1080 ������� �ͽ�������  ���������Ŀռ�
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

    // �������ź�������� gpu_res
    cuCtxPushCurrent(cuContext);
    ///YUVתBGR
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

