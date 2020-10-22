#ifndef UCNN_DNN_TYPES_H
#define UCNN_DNN_TYPES_H

namespace ucnn
{
const int kMAX_OUTPUT_LAYERS = 16;

enum PixelType
{
    RGB = 1,
    BGR = 2,
    GRAY = 3,
    RGBA = 4,

    NCHW = 5,
    YUV420SP = 6,
};

struct Shape
{
    int n = 1;
    int c = 3;
    int h = 0;
    int w = 0;
};

enum FrameworkType
{
    NCNN,
    NNIE,
    TENSORRT,
    MNN,
    TMODE,
};

enum ForwardType
{
    FORWARD_CPU,
    FORWARD_OPENCL,
    FORWARD_OPENGL,
    FORWARD_VULKAN,
};

/// for tmode
enum ModelFormat
{
    CAFFE,
    MXNET,
    TMFILE,
    SRC_TM,
};

/// for tmode
enum KernelMode
{
    KERNEL_FP32 = 0,
    KERNEL_FP16 = 1,
    KERNEL_INT8 = 2,
    KERNEL_UINT8 = 3,
};

struct ModelParams
{
    char model_prefix[256];
    char input_layer[256];
    char output_layers[kMAX_OUTPUT_LAYERS][256];
    int output_layers_num = 1;
    char model_pwd[256];
    // int max_input_size=1;
    Shape input_dims;
    int device_id = 0;
    int use_mem = 0;
    const unsigned char* net_param_bin = nullptr;
    const unsigned char* net_bin = nullptr;
    size_t net_len = 0;
    int nnie_svp_init = 0;
    PixelType input_pixel_type = BGR;

    void set_dims(int n, int c, int h, int w)
    {
        input_dims.n = n;
        input_dims.c = c;
        input_dims.h = h;
        input_dims.w = w;
    }
    int code_encryption = 0;  // 0 未加密 1加密
    FrameworkType framework_type = NCNN;
    ForwardType forward_type = FORWARD_CPU;
    int num_threads = 1;

    bool raw_output = true;  // for nnie only,  false(tensor /= 4096)

    ModelFormat model_format;
    int mxnet_epoch = 0;
    KernelMode kernel_mode = KERNEL_FP32;

    float mean_vals[3] = {0.0};        // 0 if not needed
    float std_vals[3] = {1., 1., 1.};  // 1 if not needed
};

}  // namespace ucnn

#endif  // UCNN_DNN_TYPES_H
