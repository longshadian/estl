#ifndef UCNN_CORE_MAT_H
#define UCNN_CORE_MAT_H

#include <memory>

namespace ucnn
{
class UMatData;

class Mat
{
  public:
    enum DeviceType
    {
        CPU,
        GPU,
        MMZ,
    };

    enum PlaceType
    {
        NONE,
        GRAY,
        BGR_PLANAR,
        RGB_PLANAR,
        BGR_PACKAGE,
        RGB_PACKAGE,
        YUV420SP,
        YUV422SP,
        YUV420P,
        YUV422P,
        YVU420SP,
        YVU422SP,
        YVU420P,
        YVU422P,
    };

    Mat();

    Mat(int rows, int cols, int type, DeviceType device_type, int align = 16, PlaceType place_type = NONE);

    Mat(const Mat &m);

    Mat(int rows, int cols, int type, void *data, DeviceType device_type, int align = 16, PlaceType place_type = NONE);

    Mat(int rows, int cols, int type, void *vir_add, void *phy_addr, DeviceType device_type, int align = 16,
        PlaceType place_type = NONE);

    ~Mat();

    void Create(int rows, int cols, int type, DeviceType device_type, int align = 16, PlaceType place_type = NONE);

    void Release();

    size_t ElemSize() const;

    size_t ElemSize1() const;

    int type() const;

    int depth() const;

    int channels() const;

    bool empty() const;

    size_t total() const;

    void SetHandle(int handle);

    void SetHostData(void *ptr, size_t size) const;
    void SetDeviceData(void *ptr, size_t size) const;

    ///
    /// \param need_sync if true, return after ive query,
    ///                if false(default), return without ive query
    /// \return virtual address
    void *GetHostData(bool need_sync = false) const;

    ///
    /// \param instant if true, return after ive query,
    ///                if false(default), return without ive query
    /// \return physical address on ive
    void *GetDeviceData(bool need_sync = false) const;

    void SetPlaceType(PlaceType place_type);

    PlaceType GetPlaceType() const;

    int flags;

    int rows, cols;

    int stride[3];

    int align;  // in gpu, align means step(cv::cuda::GpuMat::step)

    DeviceType device_type;

  private:
    PlaceType place_type_;

    std::shared_ptr<UMatData> data_ptr_;
};

}  // namespace ucnn

#endif  // UCNN_CORE_MAT_H