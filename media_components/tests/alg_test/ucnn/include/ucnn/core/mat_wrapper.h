#ifndef UCNN_CORE_MAT_WRAPPER_H
#define UCNN_CORE_MAT_WRAPPER_H

#include <ucnn/core/mat.h>

#ifdef BUILD_NNIE
#include <hi_comm_ive.h>
#endif

namespace cv
{
class Mat;
}

#ifdef BUILD_TENSORRT
namespace cv
{
namespace cuda
{
class GpuMat;
}
}  // namespace cv
#endif

struct hiIVE_IMAGE_S;

namespace ucnn
{
class MatWrapper
{
  public:
    MatWrapper(const Mat &mat);

    cv::Mat GetCvMat();

#ifdef BUILD_NNIE
    hiIVE_IMAGE_S GetIVEMat();
#endif

#ifdef BUILD_TENSORRT
    cv::cuda::GpuMat GetGpuMat();
#endif

  private:
    const Mat *mat_;
};
}  // namespace ucnn

#endif  // UCNN_CORE_MAT_WRAPPER_H
