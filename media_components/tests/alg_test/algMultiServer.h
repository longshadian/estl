#ifndef __ALGMOD_ALGMULTISERVER_H
#define __ALGMOD_ALGMULTISERVER_H

#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <list>
#include <cstdint>
#include <cstddef>

#include <cuda.h>
#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>
#include "common/airport_common.h"
#include "human_snap_public/human_snap_public.h"

///数据定义段
struct JpegData
{
    /// 在堆内存中的图像数据
    std::shared_ptr<  cv::Mat > data;
    /// 实际数据的大小
    int dateLen;
    /// 图像的宽高
    int imageWidth;
    int imageHeight;
    /// 图像帧ID
    uint64_t frameId;
};

/**
 * 单个最优帧的信息
 */
struct AlOutputSingleBestFrameInfo
{
    /// 图像数据
    std::shared_ptr< JpegData > bestFrameJpeg;
    /// 图像数据的算法描述信息
    std::string alInfoJson;
    /// uuid
    std::string uuid;
    /// timeStamp
    uint64_t timeStamp;
};


/**
 * 算法输出的一个人关键帧数据信息
 */
struct AlOutputKeyFrameInfo
{
    /// 人脸关健帧输出图片的数据信息
    std::shared_ptr< JpegData > keyFrameFaceOutputJpeg;
    /// 人体关健帧输出图片的数据信息
    std::shared_ptr< JpegData > keyFrameBodyOutputJpeg;

    /// 这个人对应的最优帧信息
    std::vector< std::shared_ptr< AlOutputSingleBestFrameInfo > > bestFrames;
    /// 关健帧的算法分析JSON串
    std::string alKeyFrameJson;
    /// 客行统计信息的JSON信息
    std::string passengerFlowJson;
    /// 人员ID
    uint64_t personId;
    /// UUID
    std::string alResultUuid;
    /// 时间戳, 用图片的时间戳
    uint64_t timeStamp;
};

/**
 * 算法输出最优帧信息
 */
struct AlOutputBestFrameInfo
{
    std::vector< std::shared_ptr< AlOutputSingleBestFrameInfo > > faceBestFrameInfo;
    std::vector< std::shared_ptr< AlOutputSingleBestFrameInfo > > bodyBestFrameInfo;

    /// 算法输出结构的最优帧信息
    std::string alBestFrameJson;
    /// 人员ID
    uint64_t personId;
    /// UUID
    std::string alResultUuid;
    /// 时间戳
    uint64_t timeStamp;
};



/**
 * 算法输出的数据，这些数据需要本地存储和上传到云平台。
 */
struct AlgOutPutData
{
    /// 算法输出的关键帧信息
    std::vector< std::shared_ptr< AlOutputKeyFrameInfo > > keyFrameInfo;

    /// 算法输出的最优帧信息
    std::vector< std::shared_ptr< AlOutputBestFrameInfo > > bestFrameInfo;

    /// 设备序列号
    std::string serialNumber;

    /// 工程类型
    std::string projectType;

    /// 算法工作模式
    int alWorkMode;

    int serialNumberIndex; // 设备索引

    std::shared_ptr<cv::Mat> cpu_mat;
};


#if 0
struct algMultiParam  
{
    std::shared_ptr<algGpuResource> gpu_res_;   // 临时引用
    std::int64_t time_stamp_;
    std::uint64_t frame_id_;
};
#endif


// 给算法的参数
struct algMultiDecodeParam
{
    // 解码后的视频数据，可能为空。原因是没有裸码流到来，或者nvdecoder无法解码视频。
    bool has_video_data{};
    int iW{};
    int iH{};
    unsigned char* pVdata{};
    int iVdataSize{};
    //YUV_TYPE_EN enYUVType{};
    CUcontext cuContext{};

    uint64_t gpu_res;

    int64_t frame_id{};
    int64_t frame_timestamp{};
    std::string frame_deviceno{};
};

// 多路解码server
class algMultiServer
{
public:
    algMultiServer();
    ~algMultiServer();

    int Init();
    int MulitDecode2(std::vector<algMultiDecodeParam>& argv_in,
        std::vector<std::shared_ptr<AlgOutPutData>>& result_vec_out,
        std::vector<cv::Mat>& cpu_mat_vec
    );

    void PostData(std::vector<std::shared_ptr<AlgOutPutData>> data);

    //std::shared_ptr<algGpuResource> img_nv12_scale_; // 缩放使用，如果不需要缩放，可能为空
    std::unique_ptr<ucnn::PersonSnapPublic> alg_person_snap_; // 算法句柄

    // 临时变量，可避免每次调用算法时创建
    std::vector<uniFrameInfoAlg> temp_in_frame_info_;
    std::vector<std::vector<uniBestFrameInfo>> temp_out_best_frame_info_;
    std::vector<std::vector<std::string>> temp_out_success_id_info_;

private:
    struct ImageData 
    {
        std::vector<std::shared_ptr<AlgOutPutData>> vec_;
    };

    void SaveImages(ImageData& img);
    std::size_t updateDataBase(std::shared_ptr<AlgOutPutData> algOutPutData);
    //int CheckNeedScale(VDECODE_YUVFRAME_ST& frameYUV, algGpuResource& gpu_res);
    //int CheckNeedScale2(algMultiDecodeParam& param);
    void SaveThreadRun();

    bool running_;
    std::thread save_thd_;
    std::mutex mtx_;
    std::condition_variable cond_;

    //std::list<std::shared_ptr<AlgOutPutData>> lists_;
    std::list<std::unique_ptr<ImageData>> lists_;
    std::ostringstream temp_ostm_;
};


#endif

