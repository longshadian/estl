//
//  Tensor.hpp
//  MNN
//
//  Created by MNN on 2018/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef UCNN_DNN_Tensor_hpp
#define UCNN_DNN_Tensor_hpp

#include <memory>
#include <vector>

#include "halide_runtime.h"
#include "ucnn_define.h"

namespace ucnn
{
/**
 * data container.
 * data for host tensor is saved in `host` field. its memory is allocated malloc directly.
 * data for device tensor is saved in `deviceId` field. its memory is allocated by session's backend.
 * usually, device tensors are created by engine (like net, session).
 * meanwhile, host tensors could be created by engine or user.
 */
class UCNN_PUBLIC Tensor
{
  public:
    struct InsideDescribe;

    /** dimension type used to create tensor */
    enum DimensionType
    {
        /** for tensorflow net type. uses NHWC as data format. */
        TENSORFLOW,
        /** for caffe net type. uses NCHW as data format. */
        CAFFE,
        /** for caffe net type. uses NC4HW4 as data format. */
        CAFFE_C4
    };

    /** handle type */
    enum HandleDataType
    {
        /** default handle type */
        HANDLE_NONE = 0,
        /** string handle type */
        HANDLE_STRING = 1
    };

    // remove all assignment operator
    Tensor(const Tensor& tensor) = default;
  public:
    /**
     * @brief create a tensor with dimension size and type without acquire memory for data.
     * @param dim_size   dimension size.
     * @param type      dimension type.
     */
    Tensor(int dim_size = 4, DimensionType type = CAFFE);

    Tensor(const std::vector<int>& shape, halide_type_t type, DimensionType dim_type = CAFFE, bool alloc_memory = true);

    /**
     * @brief create a tensor with same shape as given tensor.
     * @param tensor        shape provider.
     * @param type          dimension type.
     * @param alloc_memory   acquire memory for data or not.
     * @warning tensor data won't be copied.
     */
    Tensor(const Tensor* tensor, DimensionType type = CAFFE, bool alloc_memory = true);

    /** deinitializer */
    ~Tensor();

  private:
    Tensor(const Tensor&& tensor) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor& operator=(const Tensor&&) = delete;

  public:
    /**
     * @brief create tensor with shape, data type and dimension type.
     * @param shape     tensor shape.
     * @param type      data type.
     * @param dim_type   dimension type.
     * @return created tensor.
     * @warning memory for data won't be acquired. call backend's onAcquireBuffer to get memory ready.
     */
    static Tensor* CreateDevice(const std::vector<int>& shape, halide_type_t type, DimensionType dim_type = CAFFE);

    /**
     * @brief create tensor with shape and dimension type. data type is represented by `T`.
     * @param shape     tensor shape.
     * @param dim_type   dimension type.
     * @return created tensor.
     * @warning memory for data won't be acquired. call backend's onAcquireBuffer to get memory ready.
     */
    template <typename T>
    static Tensor* CreateDevice(const std::vector<int>& shape, DimensionType dim_type = CAFFE)
    {
        return CreateDevice(shape, halide_type_of<T>(), dim_type);
    }

    /**
     * @brief create tensor with shape, data type, data and dimension type.
     * @param shape     tensor shape.
     * @param type      data type.
     * @param data      data to save.
     * @param dim_type   dimension type.
     * @return created tensor.
     */
    static Tensor* Create(const std::vector<int>& shape, halide_type_t type, void* data = NULL,
                          DimensionType dim_type = CAFFE);

    /**
     * @brief create tensor with shape, data and dimension type. data type is represented by `T`.
     * @param shape     tensor shape.
     * @param data      data to save.
     * @param dim_type   dimension type.
     * @return created tensor.
     */
    template <typename T>
    static Tensor* Create(const std::vector<int>& shape, void* data = NULL, DimensionType dim_type = CAFFE)
    {
        return Create(shape, halide_type_of<T>(), data, dim_type);
    }

  public:
    /**
     * @brief for DEVICE tensor, copy data from given host tensor.
     * @param host_tensor    host tensor, the data provider.
     * @return true for DEVICE tensor, and false for HOST tensor.
     */
    bool CopyFromHostTensor(const Tensor* host_tensor);

    /**
     * @brief for DEVICE tensor, copy data to given host tensor.
     * @param host_tensor    host tensor, the data consumer.
     * @return true for DEVICE tensor, and false for HOST tensor.
     */
    bool CopyToHostTensor(Tensor* host_tensor) const;

    /**
     * @brief create HOST tensor from DEVICE tensor, with or without data copying.
     * @param device_tensor  given device tensor.
     * @param copy_data      copy data or not.
     * @return created host tensor.
     */
    static Tensor* CreateHostTensorFromDevice(const Tensor* device_tensor, bool copy_data = true);

  public:
    const halide_buffer_t& buffer() const { return buffer_; }
    halide_buffer_t& buffer() { return buffer_; }

    /**
     * @brief get dimension type.
     * @return dimension type.
     */
    DimensionType GetDimensionType() const;

    /**
     * @brief handle data type. used when data type code is halide_type_handle.
     * @return handle data type.
     */
    HandleDataType GetHandleDataType() const;

    /**
     * @brief set data type.
     * @param type data type defined in 'Type_generated.h'.
     */
    void SetType(int type);

    /**
     * @brief get data type.
     * @return data type.
     */
    inline halide_type_t GetType() const { return buffer_.type; }

    /**
     * @brief visit host memory, data type is represented by `T`.
     * @return data point in `T` type.
     */
    template <typename T>
    T* host() const
    {
        return GetHostData<T>();
    }

    /**
     * @brief GetHostData is the same with host(the function up)
     * @return data point in `T` type.
     */
    template <typename T>
    T* GetHostData() const
    {
        return (T*)buffer_.host;
    }

    template <typename T>
    T* GetDeviceData() const
    {
        return (T*)buffer_.phy_addr;
    }

    /**
     * @brief visit device memory.
     * @return device data ID. what the ID means varies between backends.
     */
    uint64_t deviceId() const { return buffer_.device; }

  public:
    int dimensions() const { return buffer_.dimensions; }

    /**
     * @brief get all dimensions' extent.
     * @return dimensions' extent.
     */
    std::vector<int> shape() const;

    /**
     * @brief calculate number of bytes needed to store data taking reordering flag into account.
     * @return bytes needed to store data
     */
    int size() const;

    /**
     * @brief calculate number of elements needed to store data taking reordering flag into account.
     * @return elements needed to store data
     */
    inline int elementSize() const { return size() / buffer_.type.bytes(); }

  public:
    inline int width() const
    {
        if (GetDimensionType() == TENSORFLOW)
        {
            return buffer_.dim[2].extent;
        }

        return buffer_.dim[3].extent;
    }
    inline int height() const
    {
        if (GetDimensionType() == TENSORFLOW)
        {
            return buffer_.dim[1].extent;
        }
        return buffer_.dim[2].extent;
    }
    inline int channel() const
    {
        if (GetDimensionType() == TENSORFLOW)
        {
            return buffer_.dim[3].extent;
        }
        return buffer_.dim[1].extent;
    }
    inline int batch() const { return buffer_.dim[0].extent; }

    // visit dimension's extent & stride
    inline int stride(int index) const { return buffer_.dim[index].stride; }
    inline int length(int index) const { return buffer_.dim[index].extent; }
    inline void SetStride(int index, int stride) { buffer_.dim[index].stride = stride; }
    inline void SetLength(int index, int length) { buffer_.dim[index].extent = length; }

  public:
    /**
     * @brief print tensor data. for DEBUG use only.
     */
    void Print() const;

  private:
    halide_buffer_t buffer_;
    struct InsideDescribe* describe_;

  private:
    friend class TensorUtils;

  private:
    void InitShape(const std::vector<int>& shape, halide_type_t type, ucnn::Tensor::DimensionType dim_type);
};

template <typename _Tp>
class Tensor_ : public Tensor
{
  public:
    Tensor_(const std::vector<int>& shape, DimensionType dim_type = CAFFE, bool alloc_memory = true)
        : Tensor(shape, halide_type_of<_Tp>(), dim_type, alloc_memory)
    {}
};

using TensorInt = Tensor_<int>;
using TensorFloat = Tensor_<float>;

std::shared_ptr<Tensor> operator*(const std::shared_ptr<TensorInt>& tensor, double a);
std::shared_ptr<Tensor> operator*(const std::shared_ptr<TensorInt>& tensor, int a);

std::shared_ptr<Tensor> operator*(const std::shared_ptr<TensorFloat>& tensor, double a);
std::shared_ptr<Tensor> operator*(const std::shared_ptr<TensorFloat>& tensor, int a);

std::shared_ptr<Tensor> operator/(const std::shared_ptr<TensorInt>& tensor, double a);
std::shared_ptr<Tensor> operator/(const std::shared_ptr<TensorInt>& tensor, int a);

std::shared_ptr<Tensor> operator/(const std::shared_ptr<TensorFloat>& tensor, double a);
std::shared_ptr<Tensor> operator/(const std::shared_ptr<TensorFloat>& tensor, int a);

std::shared_ptr<Tensor> operator*=(const std::shared_ptr<Tensor>& tensor, double a);
std::shared_ptr<Tensor> operator/=(const std::shared_ptr<Tensor>& tensor, double a);

}  // namespace ucnn

#endif /* UCNN_DNN_Tensor_hpp */
