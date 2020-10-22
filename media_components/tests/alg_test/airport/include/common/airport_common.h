//
// Created by pengguanhai on 2020/4/16.
//

#ifndef AIRPORT_TRACKING_AIRPORT_COMMON_H
#define AIRPORT_TRACKING_AIRPORT_COMMON_H

#include <stdint.h>
#include <memory>
#include <vector>

/// \brief 数据存放路径
enum uniStorageType
{
    DATA_IN_CPU = 0,  ///< 数据存放在CPU端
    DATA_IN_GPU = 1,  ///< 数据存放在GPU端（cuda）
    DATA_IN_MMZ = 2   ///< 数据存放在MMZ端（hisi）
};

/// \brief 图像格式类型
enum uniPixelFormat
{
    IsGRAY = 0,           ///< 灰度图像
    RGB_PLANAR = 1,       ///< RRRGGGBBB 数据格式
    BGR_PLANAR = 2,       ///< BBBGGGRRR 数据格式
    RGB_PACKAGE = 3,      ///< RGBRGBRGB 数据格式
    BGR_PACKAGE = 4,      ///< BGRBGRBGR 数据格式
    YUV_420SP = 5,        ///< YYYYYYYYUVUV 数据格式
    YUV_420P = 6,         ///< YYYYYYYYUUVV 数据格式
    ImageFormatBUTT = 7,  ///< 异常
};

/// \brief 视频图像帧信息
struct uniFrameInfoAlg
{
    uint64_t frame_id;                  ///< 图像数据ID(帧号)
    uint16_t frame_width;               ///< 图像宽度信息
    uint16_t frame_height;              ///< 图像高度信息
    uint16_t frame_stride;              ///< 图像对齐后信息
    uint64_t time_stamp;                ///< 图像数据时间戳
    uniPixelFormat frame_pixel_format;  ///< 图像数据格式
    uniStorageType frame_storage_type;  ///< 图像数据存储类型

    uint64_t VirAddress;  ///< 输入图像数据虚拟地址（默认数据为连续存储）
    uint64_t PhyAddress;  ///< 输入图像数据物理地址（默认数据为连续存储）
};

/// \brief 点
template <class T>
struct uniPoint_
{
    T x;  ///< 横坐标参数
    T y;  ///< 纵坐标参数
};

using uniPoint = uniPoint_<int>;
using uniPoint2f = uniPoint_<float>;

/// \brief 矩形框, 上下左右边界信息
template <class T>
struct uniRect_
{
    T xLeft;    /// 左边框信息
    T yTop;     /// 上边框信息
    T xRight;   /// 右边框信息
    T yBottom;  /// 下边框信息
};

using uniRect = uniRect_<int>;
using uniRect2f = uniRect_<float>;

/// \brief 尺度
template <class T>
struct uniSize_
{
    T height;
    T width;
};
using uniSize = uniSize_<int>;
using uniSize2f = uniSize_<float>;

/// \brief 人员结构化信息
struct uniHumanInfo
{
    uint64_t human_id = -1;                       ///< 行人ID信息
    std::string human_id_uuid;                       ///< 行人ID信息 uuid
    uniRect2f head_rect = uniRect2f{0, 0, 0, 0};  ///< 人头外接框信息
    uniRect2f body_rect = uniRect2f{0, 0, 0, 0};  ///< 人体外接框信息
    uniPoint2f replace_point = uniPoint2f{0, 0};  ///< 人体相对替代点信息

    float face_quality = 0.f;  ///< 人脸质量分数
    float body_quality = 0.f;  ///< 人体质量分数

    float head_rect_score = 0.f;  ///< 人头框检测分数
    float body_rect_score = 0.f;  ///< 人体框检测分数
};

/// \brief 关键帧图像智能指针类
class SmartFrameInfoClass
{
  public:
    SmartFrameInfoClass(const uint64_t frame_id, const uint64_t time_stamp, const uint16_t frame_width,
                        const uint16_t frame_height, const uint16_t frame_stride,
                        const uniPixelFormat frame_pixel_format, const uniStorageType frame_storage_type);
    ~SmartFrameInfoClass();
    uniFrameInfoAlg frameInfo;
};

/// \brief 单个目标最优帧数据
struct uniBestFrameSingle
{
    uint64_t person_id;                                 ///< 图像对应ID
    std::string human_id_uuid;                             ///< 行人ID信息 uuid
    float quality;                                      ///< 图像质量信息
    uniRect2f rect;                                     ///< 感兴趣区域坐标
    std::shared_ptr<SmartFrameInfoClass> best_img_ptr;  ///< 最优帧图像
};

/// \brief 目标最优帧数据
struct uniBestFrameInfo
{
    std::vector<uniBestFrameSingle> best_frame_face_info;  ///< 人脸最优帧数据
    std::vector<uniBestFrameSingle> best_frame_body_info;  ///< 人体最优帧数据
};

struct uniConfigParam
{
    //    bool full_flag = false;        /// 最优帧为全图备份标志（全图 ||
    //    扣取）由于硬件资源不足DMA备份，所以目前只支持抠图模式
    int track_times_th = 3;             /// 最小跟踪帧数
    int upload_interval = 10;           /// 重抓（上传）间隔
    int best_frame_num = 3;             /// 最优帧数量
    float best_face_quality_th = 0.3f;  /// 最优帧人脸质量阈值
    float best_body_quality_th = 0.3f;  /// 最优帧人体质量阈值
};

#endif  // AIRPORT_TRACKING_AIRPORT_COMMON_H
