#include "algMultiServer.h"

#include "../common/utility.h"
#include "../common/console_log.h"

#include "NvCodec/Utils/NvCodecUtils.h"
#include "NvCodec/Utils/ColorSpace.h"

#define EASYLOG_INFO CONSOLE_LOG_INFO
#define EASYLOG_WARNING CONSOLE_LOG_WARN

#define MAX_ALG_IMAGE_WIDTH int(1920)
#define  MAX_ALG_IMAGE_HEIGHT int(1080)

static std::string Get_FILES_STORAGE_PATH()
{
    return "/home/bolan/works/pic_alg/";
}


#if 0
/*
static void YUVFrameNv12ToBgr(VDECODE_YUVFRAME_ST& yuv_frame, algGpuResource& gpu_res)
{
    cuCtxPushCurrent(yuv_frame.cuContext);
    ///YUV转BGR
    Nv12ToBgr(yuv_frame.pVdata, (int)yuv_frame.iW, (uint8_t*)gpu_res.gpu_buffer_,
              MAX_ALG_IMAGE_WIDTH, MAX_ALG_IMAGE_WIDTH, MAX_ALG_IMAGE_HEIGHT);
    cuCtxPopCurrent(NULL);
}
*/


static void BestFrameAlInfo_To_Json(const uniBestFrameSingle& bestFrameSingle, int64_t currentTimeMS,
    int type, int index, std::string& out_str)
{
    JsonGenerator jsonGenerator;
    jsonGenerator.addMember("personId", bestFrameSingle.person_id);
    if (type == 0) {
        jsonGenerator.addMember("type", "face");
    } else if (type == 1) {
        jsonGenerator.addMember("type", "body");
    } else {
        jsonGenerator.addMember("type", "body");
    }

    jsonGenerator.addMember("index", index);
    jsonGenerator.addMember("timeStamp", currentTimeMS);
    jsonGenerator.addMember("imageWidth", bestFrameSingle.best_img_ptr->frameInfo.frame_width);
    jsonGenerator.addMember("imageHeight", bestFrameSingle.best_img_ptr->frameInfo.frame_height);
    jsonGenerator.addMember("quality", bestFrameSingle.quality);

    jsonGenerator.startObject("box");
    jsonGenerator.addMember("xLeft", bestFrameSingle.rect.xLeft);
    jsonGenerator.addMember("yTop", bestFrameSingle.rect.yTop);
    jsonGenerator.addMember("xRight", bestFrameSingle.rect.xRight);
    jsonGenerator.addMember("yBottom", bestFrameSingle.rect.yBottom);
    jsonGenerator.finishObject();

    out_str = jsonGenerator.toString();
}

static void BestFrameHead_To_Json(const std::string& uuid, uint64_t personId,
    uint64_t timeStamp, std::string& json)
{
    JsonGenerator jsonGenerator;
    jsonGenerator.addMember("uuid", uuid);
    jsonGenerator.addMember("personId", personId);

    jsonGenerator.startObject("system");
    jsonGenerator.addMember("projectType", PROJECT_TYPE);
    // TODO 以下这些字段有用吗？？
    // jsonGenerator.addMember("sn", m_ostrDevicekey);
    // jsonGenerator.addMember("alVersion", AL_VERSION);
    // jsonGenerator.addMember("boardType", BOARD_TYPE);
    jsonGenerator.addMember("time", timeStamp);
    jsonGenerator.finishObject();

    jsonGenerator.startObject("alParameters");
    {
        jsonGenerator.addMember("workMode", 2);//工作模型2 抓拍模式
        jsonGenerator.addMember("topN", 3);///最优帧top3
    }
    jsonGenerator.finishObject();
    json = std::move(jsonGenerator.toString());
}
#endif

static void BestFaceOrBody(const std::vector<uniBestFrameSingle>& bestFaceFrames,
    const std::string& singlePersonUuid,
    int64_t currentTimeMS,
    int face_or_body,
    uint64_t* person_id,
    std::vector<std::shared_ptr<AlOutputSingleBestFrameInfo>>& out)
{
    int index = 0;
    for (const uniBestFrameSingle& bestFrame : bestFaceFrames) {
        const SmartFrameInfoClass* best_img_ptr = bestFrame.best_img_ptr.get();
        if (!bestFrame.best_img_ptr)
            continue;
        if (best_img_ptr->frameInfo.frame_height > static_cast<uint16_t>(MAX_ALG_IMAGE_HEIGHT) 
            || best_img_ptr->frameInfo.frame_width > static_cast<uint16_t>(MAX_ALG_IMAGE_WIDTH)) {
            continue;
        }
        *person_id = bestFrame.person_id;

        auto single_info = std::make_shared<AlOutputSingleBestFrameInfo>();
        single_info->bestFrameJpeg = std::make_shared<JpegData>();
        single_info->bestFrameJpeg->data = std::make_shared<cv::Mat>();
        ///将图片数据存入本地
        {
            if (1) {
                cv::cuda::GpuMat best_mat_gpu(best_img_ptr->frameInfo.frame_height,
                    best_img_ptr->frameInfo.frame_width, CV_8UC3,
                    (void*)best_img_ptr->frameInfo.PhyAddress,
                    best_img_ptr->frameInfo.frame_stride);
                best_mat_gpu.download(*single_info->bestFrameJpeg->data);
            }

            if (0) {
                // 测试使用，查看算法生产的图片是否正确。
                static int cnt = 0;
                ++cnt;
                std::ostringstream ostm;
                ostm << "/home/bolan/works/tmp_data/" << singlePersonUuid <<"_" << face_or_body 
                    << "_" << index << "_" << cnt<< ".jpeg";
                std::string s = ostm.str();
                cv::imwrite(s.c_str(),  *(single_info->bestFrameJpeg->data));
            }
        }

        single_info->bestFrameJpeg->frameId = best_img_ptr->frameInfo.frame_id;
        single_info->bestFrameJpeg->imageWidth = best_img_ptr->frameInfo.frame_width;
        single_info->bestFrameJpeg->imageHeight = best_img_ptr->frameInfo.frame_height;
        single_info->uuid = singlePersonUuid;
        single_info->timeStamp = best_img_ptr->frameInfo.time_stamp / 1000;
        /// 生成算法描述信息
        //BestFrameAlInfo_To_Json(bestFrame, currentTimeMS, face_or_body, index, single_info->alInfoJson);
        out.push_back(single_info);
        index++;
    }
}

static std::shared_ptr<AlgOutPutData>
    storageAndUpdateDataBase(const std::vector<uniBestFrameInfo>&  bestFrameInfo)
{
    std::shared_ptr<AlgOutPutData> outputParam = nullptr;
    //outputParam->serialNumber = m_ostrDevicekey;///;//开头信息 先写死
    // 人体关健帧以及人脸关健帧保存
    //outputParam->projectType = PROJECT_TYPE;

    int64_t curMs = comm::unix_time_milliseconds();
    // 最优帧进行JPEG编码
    for (const uniBestFrameInfo& frame : bestFrameInfo) {
        // 没有图片，就不保存json。
        if (frame.best_frame_face_info.empty() && frame.best_frame_body_info.empty())
            continue;

        auto best_frame_info = std::make_shared<AlOutputBestFrameInfo>();
        // TODO personId ??
        uint64_t personId = 0;

        /// 人脸最优帧
        static int x = 0;
        //std::string singlePersonUuid = generateUUID();
        std::string singlePersonUuid = std::to_string(comm::unix_time_microseconds());
        singlePersonUuid += "_" + std::to_string(++x);
        BestFaceOrBody(frame.best_frame_face_info, singlePersonUuid, curMs, 0, &personId, best_frame_info->faceBestFrameInfo);

        /// 人体最优帧
        BestFaceOrBody(frame.best_frame_body_info, singlePersonUuid, curMs, 1, &personId, best_frame_info->bodyBestFrameInfo);

        best_frame_info->alResultUuid = singlePersonUuid;
        best_frame_info->personId = personId;
        best_frame_info->timeStamp = static_cast<std::uint64_t>(curMs);
        /// 将单个人的人脸最优帧和人体最优帧的算法数据JSON化
        //BestFrameHead_To_Json(singlePersonUuid, personId, curMs, best_frame_info->alBestFrameJson);

        if (!outputParam)
            outputParam = std::make_shared< AlgOutPutData >();
        outputParam->bestFrameInfo.push_back(best_frame_info);
    }
    return outputParam;
}

static void YUVFrameNv12ToBgr(algMultiDecodeParam& param)
{
    cuCtxPushCurrent(param.cuContext);
    ///YUV转BGR
    Nv12ToBgr(param.pVdata, (int)param.iW, (uint8_t*)param.gpu_res,
              MAX_ALG_IMAGE_WIDTH, MAX_ALG_IMAGE_WIDTH, MAX_ALG_IMAGE_HEIGHT);
              /*
    Nv12ToColor32<BGRA32>((uint8_t*)param.pVdata, (int)param.iW, (uint8_t*)param.gpu_res,
              MAX_ALG_IMAGE_WIDTH, MAX_ALG_IMAGE_WIDTH, MAX_ALG_IMAGE_HEIGHT);
              */
    cuCtxPopCurrent(NULL);
}


/************************************************************************/
/* class algMultiServer                                                 */
/************************************************************************/

algMultiServer::algMultiServer()
#if 0
    : img_nv12_scale_() 
    , alg_person_snap_()
    , temp_in_frame_info_()
    , temp_out_best_frame_info_()
    , temp_out_success_id_info_()
    , running_()
    , save_thd_()
    , mtx_()
    , cond_()
    , lists_()
    , temp_ostm_()
#endif
    : alg_person_snap_()
    , temp_in_frame_info_()
    , temp_out_best_frame_info_()
    , temp_out_success_id_info_()

    , running_()
    , save_thd_()
    , mtx_()
    , cond_()
    , lists_()
    , temp_ostm_()
{
}

algMultiServer::~algMultiServer()
{
    running_ = false;
    if (save_thd_.joinable())
        save_thd_.join();
}

int algMultiServer::Init()
{
    std::string model_prefix = "/home/bolan/works/PAS/models";
    std::string verify_name = "/home/bolan/works/PAS/tools/verify_X86_airport.lic";
    std::string eth_name = "enp9s0f0";
    EASYLOG_INFO << "alg conf: modle_prefix: " << model_prefix;
    EASYLOG_INFO << "alg conf: verify_name: " << verify_name;
    EASYLOG_INFO << "alg conf: eth_name: " << eth_name;

    // 创建框框
    const int gpuid = 0;
    const int channel_count = 1;
    ucnn::Rect roibox(0, 0, MAX_ALG_IMAGE_WIDTH, MAX_ALG_IMAGE_HEIGHT);
    std::vector<ucnn::Rect> roi_multi(channel_count, roibox);
    alg_person_snap_ = std::make_unique<ucnn::PersonSnapPublic>(model_prefix
        , roi_multi, verify_name, eth_name, gpuid);
    //指定显卡号
    EASYLOG_INFO << "init decoder gpuid: " << gpuid;

    // 启动保存线程
#if 1
    std::thread temp(std::bind(&algMultiServer::SaveThreadRun, this));
    std::swap(temp, save_thd_);
#endif
    return 0;
}

int algMultiServer::MulitDecode2(std::vector<algMultiDecodeParam>& alg_params,
    std::vector<std::shared_ptr<AlgOutPutData>>& result_vec_out)
{
    temp_in_frame_info_.clear();
    temp_in_frame_info_.resize(alg_params.size());
    temp_out_success_id_info_.resize(alg_params.size());
    temp_out_best_frame_info_.clear();

    // 填充gpu_buffer
    for (size_t i = 0; i != alg_params.size(); ++i) {
        algMultiDecodeParam& param = alg_params[i];
#if 0
        if (param.has_video_data) {
            param.gpu_res_has_data = true;
            // 如果有视频数据，进行缩放，yuv->bgr转换，转换后的数据放在gpu_res中
            int ret = 0;
            if (param.iW != MAX_ALG_IMAGE_WIDTH || param.iH != MAX_ALG_IMAGE_HEIGHT) {
                ret = CheckNeedScale2(param);
                //EASYLOG_INFO << "CheckNeedScale: " << frameYUV.iW << " " << frameYUV.iH;
            } else {
                ///YUV转BGR
                YUVFrameNv12ToBgr(param);
            }
            (void)ret;
        } else {
            // 如果没有视频数据，清理gpu_res中的内容，防止算法获取到旧的数据
            if (param.gpu_res_has_data) {
                // TODO 20200921 此处代码先关闭，开启后可能导致算法出现全灰图。
                //      单线程执行按理说，不应该出现这种情况。
                //      此bug需要修复，原因还没找到。
                //cudaMemset((uint8_t*)param.gpu_res->gpu_buffer_, 128, param.gpu_res->Size());
                param.gpu_res_has_data = false;
                //EASYLOG_INFO << "----------------- clean pic 2";
            }
        }
#endif

        ///YUV转BGR
        YUVFrameNv12ToBgr(param);
        if (param.frame_id > 0 && 
            param.frame_id % 100 == 0) {
            std::vector<char> img_buffer{};
            img_buffer.resize(param.iVdataSize);
            cudaMemcpy(img_buffer.data(), (void*)param.pVdata, param.iVdataSize, cudaMemcpyDeviceToHost);

            char buf[64] = { 0 };
            snprintf(buf, sizeof(buf), "%s/bgr_%d.yuv", "/home/bolan/works/pic", (int)param.frame_id);
            comm::SaveFile(buf, img_buffer.data(), img_buffer.size());
        }

        // 配置算法参数
        uniFrameInfoAlg& finfo = temp_in_frame_info_[i];
        finfo.PhyAddress = (uint64_t)param.gpu_res; // 显存物理地址
        finfo.frame_id = param.frame_id;
        finfo.frame_width = MAX_ALG_IMAGE_WIDTH;
        finfo.frame_height = MAX_ALG_IMAGE_HEIGHT;
        finfo.frame_stride = 5760;
        finfo.time_stamp = comm::unix_time_milliseconds();
        finfo.frame_storage_type = DATA_IN_GPU;
        finfo.frame_pixel_format = BGR_PACKAGE;
    }

#if 0
// 本地调试算法使用
    if (0) {
       static int n = 0;
       ++n;
       int ret = 0;
       (void)ret;
       //int gpuid = algGlobalIns::Get()->GpuID_;
        if (1) {
            ///测试程序 将送入算法的数据存本地 查看是否正常
           int frame_id = alg_params[0].frame_id;
           cv::cuda::GpuMat source_gpu(1080, 1920,CV_8UC3, (void*)(temp_in_frame_info_[0].PhyAddress));
           cv::Mat source_cpu;
           source_gpu.download(source_cpu);
           std::ostringstream ostm;
           ostm << "/home/bolan/works/pic/x_" << frame_id << ".jpg";
           std::string pic_name = ostm.str();
           EASYLOG_INFO << pic_name;
           //cv::cuda::setDevice(gpuid);
            //EASYLOG_INFO << "----------------- 4444 set cv gpuid: " << gpuid;
           cv::imwrite(pic_name, source_cpu);
       }
       if (0) {
           cv::Mat source_cpu(1080, 1920, CV_8UC3);
           cudaMemcpy(source_cpu.data, (void*)(temp_in_frame_info_[0].PhyAddress), 1920 * 3 * 1080, cudaMemcpyDeviceToHost);
           std::ostringstream ostm;
           ostm << "/home/bolan/works/pic/x_" << n << ".jpg";
           std::string xss = ostm.str();
           ret = cv::imwrite(xss, source_cpu);
        }
    }
#endif

    //时间测试
    comm::MicrosecondTimer alg_timer;
    alg_timer.Start();
    // 算法入口, 多路视频，调用一次算法。
    uniConfigParam config_param{};
    //config_param.best_face_quality_th = 0.0f;

#if 1
    //std::vector<std::vector<uniHumanInfoAlg>> real_time_human_info_multi = alg_person_snap_->snap(temp_in_frame_info_, config_param, temp_out_success_id_info_, temp_out_best_frame_info_);
    std::vector<std::vector<uniHumanInfo>> real_time_human_info_multi = alg_person_snap_->snap(temp_in_frame_info_, config_param, temp_out_success_id_info_, temp_out_best_frame_info_);
#endif
    alg_timer.Stop();
    EASYLOG_INFO << "ptimer alg cost: " << alg_timer.GetMilliseconds();

#if 1
    // 生成图片
    assert(temp_out_best_frame_info_.size() == alg_params.size());
    for (std::size_t i = 0; i != temp_out_best_frame_info_.size(); ++i) {
        const auto& best_frame = temp_out_best_frame_info_[i];
        if (best_frame.empty()) {
            continue;
        }
        auto p = storageAndUpdateDataBase(best_frame);
        if (!p) {
            continue;
        }
        p->serialNumber = alg_params[i].frame_deviceno;
        result_vec_out.push_back(p);
    }

    if (!result_vec_out.empty()) {
        //打印算法耗时
        //auto use_time = alg_timer.Delta();
        //EASYLOG_INFO << "ptimer alg cost batch: " << alg_params.size() << " cost: " << (use_time/1000) << "." << (use_time%1000) << " ms";
        EASYLOG_INFO << "ptimer alg cost: " << int(result_vec_out.empty()) <<" batch: " << alg_params.size() << " cost: " << alg_timer.GetMilliseconds();
    }
#endif
    return 0;
}

void algMultiServer::PostData(std::vector<std::shared_ptr<AlgOutPutData>> data)
{
    auto p = std::make_unique<ImageData>();
    p->vec_ = std::move(data);
    {
        std::lock_guard<std::mutex> lk(mtx_);
        lists_.emplace_back(std::move(p));

        if (1) {
            //TODO 保护代码，解码、算法产生的图片数据非常快，sqlite保存非常慢。
            // 会导致此队列无限增长, 耗尽内存。这里限制最多保存N条，超过限制后，替换掉旧数据。
            if (lists_.size() > 1000) {
                lists_.pop_front();
                EASYLOG_WARNING << "too many image data in queue. size() > 1000. discard old image data";
            }
        }
        cond_.notify_all();
    }
}

#if 0
int algMultiServer::CheckNeedScale(VDECODE_YUVFRAME_ST& frameYUV, algGpuResource& gpu_res)
{
    ///假如图像长和宽 不是1920 和 1080 的情况下 就进行缩放  存入刚申请的空间
    if (!img_nv12_scale_) {
        auto p = std::make_shared<algGpuResource>();
        cuCtxPushCurrent(frameYUV.cuContext);
        cuMemAlloc(&p->gpu_buffer_, MAX_ALG_IMAGE_WIDTH * MAX_ALG_IMAGE_HEIGHT * 3 / 2);
        cuCtxPopCurrent(NULL);

        // TODO check 分配失败
        img_nv12_scale_ = p;
    }
    if (!img_nv12_scale_) {
        EASYLOG_WARNING << "img_nv12_scale failure.";
        return -1;
    }
    cuCtxPushCurrent(frameYUV.cuContext);
    ResizeNv12((uint8_t*)img_nv12_scale_->gpu_buffer_, MAX_ALG_IMAGE_WIDTH, MAX_ALG_IMAGE_WIDTH, MAX_ALG_IMAGE_HEIGHT,
        frameYUV.pVdata, (int)frameYUV.iW, (int)frameYUV.iW, (int)frameYUV.iH, NULL);
    cuCtxPopCurrent(NULL);

    // 复制缩放后的数据至 gpu_res
    cuCtxPushCurrent(frameYUV.cuContext);
    ///YUV转BGR
    Nv12ToBgr((uint8_t*)img_nv12_scale_->gpu_buffer_, (int)MAX_ALG_IMAGE_WIDTH, (uint8_t*)gpu_res.gpu_buffer_,
              MAX_ALG_IMAGE_WIDTH, MAX_ALG_IMAGE_WIDTH, MAX_ALG_IMAGE_HEIGHT);
    cuCtxPopCurrent(NULL);

    return 0;
}

int algMultiServer::CheckNeedScale2(algMultiDecodeParam& param)
{
    ///假如图像长和宽 不是1920 和 1080 的情况下 就进行缩放  存入刚申请的空间
    if (!img_nv12_scale_) {
        auto p = std::make_shared<algGpuResource>();
        cuCtxPushCurrent(param.cuContext);
        cuMemAlloc(&p->gpu_buffer_, MAX_ALG_IMAGE_WIDTH * MAX_ALG_IMAGE_HEIGHT * 3 / 2);
        cuCtxPopCurrent(NULL);

        // TODO check 分配失败
        img_nv12_scale_ = p;
    }
    if (!img_nv12_scale_) {
        EASYLOG_WARNING << "img_nv12_scale failure.";
        return -1;
    }
    cuCtxPushCurrent(param.cuContext);
    ResizeNv12((uint8_t*)img_nv12_scale_->gpu_buffer_, MAX_ALG_IMAGE_WIDTH, MAX_ALG_IMAGE_WIDTH, MAX_ALG_IMAGE_HEIGHT,
        param.pVdata, (int)param.iW, (int)param.iW, (int)param.iH, NULL);
    cuCtxPopCurrent(NULL);

    // 复制缩放后的数据至 gpu_res
    cuCtxPushCurrent(param.cuContext);
    ///YUV转BGR
    Nv12ToBgr((uint8_t*)img_nv12_scale_->gpu_buffer_, (int)MAX_ALG_IMAGE_WIDTH, (uint8_t*)param.gpu_res->gpu_buffer_,
              MAX_ALG_IMAGE_WIDTH, MAX_ALG_IMAGE_WIDTH, MAX_ALG_IMAGE_HEIGHT);
    cuCtxPopCurrent(NULL);

    return 0;
}
#endif

void algMultiServer::SaveThreadRun()
{
    running_ = true;
    std::unique_ptr<ImageData> p;
    const std::chrono::seconds sec(1);

    size_t size = 0;
    while (running_) {
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cond_.wait_for(lk, sec, [this] { return !lists_.empty(); });
            if (lists_.empty())
                continue;
            p = std::move(lists_.front());
            lists_.pop_front();
            size = lists_.size();
        }
        if (p) {
            SaveImages(*p);
            p = nullptr;
            EASYLOG_INFO << "image data size: " << size;
        }
    }
}

void algMultiServer::SaveImages(ImageData& img)
{
    if (1) {
        // 开启打印多路视频，算法处理结果
        temp_ostm_.str("");
        std::size_t cnt = 0;
        temp_ostm_ << "[";
        for (auto& p : img.vec_) {
            cnt = updateDataBase(p);
            temp_ostm_ << p->serialNumber << "_" << cnt << ", ";
        }
        temp_ostm_ <<"] ";
        EASYLOG_INFO << "alg batch: " << img.vec_.size() << " " << temp_ostm_.str();
    } else {
        for (auto& p : img.vec_) {
            updateDataBase(p);
        }
    }
}

#if 0
std::size_t algMultiServer::updateDataBase(std::shared_ptr<AlgOutPutData> algOutPutData)
{
    /// 最优帧信息存储和数据库更新
    //EASYLOG_INFO << "----------> bestFrameInfo " << algOutPutData->bestFrameInfo.size();
    if (algOutPutData->bestFrameInfo.empty()) {
        return 0;
    }

    comm::MicrosecondTimer total_timer;
    total_timer.Start();

    auto tnow = std::time(nullptr);
    std::vector< std::shared_ptr< DB_BestFrameData > > bestData;
    auto timestamp_str = comm::Localtime_YYYYMMDD_HHMMSS(&tnow);

    int index = 0;
    std::ostringstream ostm{};
    for (const auto& bestFrameInfo : algOutPutData->bestFrameInfo) {
        /// 人脸最优帧
        index = 0;
        for (const auto& faceBestFrame : bestFrameInfo->faceBestFrameInfo)
        {
            //if (!faceBestFrame->bestFrameJpeg.unique()) { continue; }

            ostm.str("");
            ostm << Get_FILES_STORAGE_PATH() << timestamp_str 
                << "_" << faceBestFrame->uuid 
                << "_" << bestFrameInfo->personId
                << "_" << index
                << "_"  << faceBestFrame->bestFrameJpeg->frameId
                << "_" << "bestface.jpeg"
            ;
            std::string faceJpegName = ostm.str();
            index += 1;
            //if (faceBestFrame->bestFrameJpeg.unique())
            {
                commex::MicrosecondTimer face_timer;
                face_timer.Start();
                cv::imwrite(faceJpegName, *(faceBestFrame->bestFrameJpeg->data));
                face_timer.Stop();
                EASYLOG_INFO << "ptimer face pic cost: " << face_timer.GetMilliseconds();

                EASYLOG_INFO << "save faceJpegName: " << faceJpegName;
                /// 信息更新到数据库中
                auto bestFrameData = std::make_shared< DB_BestFrameData >();
                bestFrameData->filePath = faceJpegName;
                bestFrameData->timeStamp = faceBestFrame->timeStamp;
                bestFrameData->uuID = bestFrameInfo->alResultUuid;
                bestFrameData->attribute = faceBestFrame->alInfoJson;
                bestFrameData->fileType = PIC_FILE_TYPE;
                bestData.push_back(bestFrameData);
            }
        }
        /// 人体最优帧
        index = 0;
        for (const auto& bodyBestFrame : bestFrameInfo->bodyBestFrameInfo)
        {
            //if (!bodyBestFrame->bestFrameJpeg.unique()) { continue; }
            ostm.str("");
            ostm << Get_FILES_STORAGE_PATH() << timestamp_str
                << "_" << bodyBestFrame->uuid
                << "_"  << bestFrameInfo->personId
                << "_" << index
                << "_" << bodyBestFrame->bestFrameJpeg->frameId
                << "_" << "bestbody.jpeg";
            std::string bodyJpegName = ostm.str();
            index += 1;
            //if (bodyBestFrame->bestFrameJpeg.unique()) 
            {
                commex::MicrosecondTimer body_timer;
                body_timer.Start();
                cv::imwrite(bodyJpegName, *(bodyBestFrame->bestFrameJpeg->data));
                body_timer.Stop();
                EASYLOG_INFO << "ptimer body pic cost: " << body_timer.GetMilliseconds();

                EASYLOG_INFO << "save bodyJpegName: " << bodyJpegName;
                auto bestFrameData = std::make_shared< DB_BestFrameData >();
                bestFrameData->filePath = bodyJpegName;
                bestFrameData->timeStamp = bodyBestFrame->timeStamp;
                bestFrameData->uuID = bodyBestFrame->uuid;
                bestFrameData->attribute = bodyBestFrame->alInfoJson;
                bestFrameData->fileType = PIC_FILE_TYPE;
                bestData.push_back(bestFrameData);
            }

        }
        /// 最优帧HEAD信息写入文件,至少有1张face or body
        if (bestFrameInfo->alBestFrameJson.length() != 0
            && (!bestFrameInfo->faceBestFrameInfo.empty() || !bestFrameInfo->bodyBestFrameInfo.empty()) 
            )
        {
            ostm.str("");
            ostm << Get_FILES_STORAGE_PATH() << timestamp_str
                << "_"  << bestFrameInfo->alResultUuid
                << "_" << bestFrameInfo->personId
                << "_" << "bestFrameHead.json";
            std::string bestFrameHeadJsonFileName = ostm.str();
            std::ofstream outFile1(bestFrameHeadJsonFileName, std::ios::out | std::ios::binary);
            if (outFile1 && (bestFrameInfo->alBestFrameJson.length() != 0))
            {
                outFile1.write((char*)bestFrameInfo->alBestFrameJson.c_str(), bestFrameInfo->alBestFrameJson.length());
                outFile1.close();
                auto bestFrameJsonData = std::make_shared< DB_BestFrameData >();
                bestFrameJsonData->filePath = bestFrameHeadJsonFileName;
                bestFrameJsonData->timeStamp = bestFrameInfo->timeStamp;
                bestFrameJsonData->uuID = bestFrameInfo->alResultUuid;
                bestFrameJsonData->attribute = "bestFrameJsonHead";
                bestFrameJsonData->fileType = JSON_FILE_TYPE;
                bestData.push_back(bestFrameJsonData);
            } else {
                outFile1.close();
                EASYLOG_WARNING << "open file failure error filename: " << bestFrameHeadJsonFileName << " length: " << bestFrameInfo->alBestFrameJson.length();
            }
        }
    }
    //
    if (bestData.empty() == false) {
        commex::MicrosecondTimer db_timer;
        db_timer.Start();
        Sqlite3Db::Sqlite3Db_getInstance()->SqliteDb_setRecord(bestData, SQLITE3_TB_NMAE_BEST);
        db_timer.Stop();
        EASYLOG_INFO << "ptimer db cost: " << db_timer.GetMilliseconds();
    }
    total_timer.Stop();
    EASYLOG_INFO << "ptimer total cost: " << total_timer.GetMilliseconds();

    return bestData.size();
}
#endif

std::size_t algMultiServer::updateDataBase(std::shared_ptr<AlgOutPutData> algOutPutData)
{
    /// 最优帧信息存储和数据库更新
    //EASYLOG_INFO << "----------> bestFrameInfo " << algOutPutData->bestFrameInfo.size();
    if (algOutPutData->bestFrameInfo.empty()) {
        return 0;
    }

    comm::MicrosecondTimer total_timer;
    total_timer.Start();

    auto tnow = std::time(nullptr);
    auto timestamp_str = comm::Localtime_YYYYMMDD_HHMMSS(&tnow);

    int index = 0;
    std::ostringstream ostm{};
    for (const auto& bestFrameInfo : algOutPutData->bestFrameInfo) {
        /// 人脸最优帧
        index = 0;
        for (const auto& faceBestFrame : bestFrameInfo->faceBestFrameInfo) {
            ostm.str("");
            ostm << Get_FILES_STORAGE_PATH() << timestamp_str 
                << "_" << faceBestFrame->uuid 
                << "_" << bestFrameInfo->personId
                << "_" << index
                << "_"  << faceBestFrame->bestFrameJpeg->frameId
                << "_" << "bestface.jpeg"
            ;
            std::string faceJpegName = ostm.str();
            index += 1;
            //if (faceBestFrame->bestFrameJpeg.unique())
            {
                comm::MicrosecondTimer face_timer;
                face_timer.Start();
                cv::imwrite(faceJpegName, *(faceBestFrame->bestFrameJpeg->data));
                face_timer.Stop();
                EASYLOG_INFO << "ptimer face pic cost: " << face_timer.GetMilliseconds();

                EASYLOG_INFO << "save faceJpegName: " << faceJpegName;
            }
        }
        /// 人体最优帧
        index = 0;
        for (const auto& bodyBestFrame : bestFrameInfo->bodyBestFrameInfo)
        {
            //if (!bodyBestFrame->bestFrameJpeg.unique()) { continue; }
            ostm.str("");
            ostm << Get_FILES_STORAGE_PATH() << timestamp_str
                << "_" << bodyBestFrame->uuid
                << "_"  << bestFrameInfo->personId
                << "_" << index
                << "_" << bodyBestFrame->bestFrameJpeg->frameId
                << "_" << "bestbody.jpeg";
            std::string bodyJpegName = ostm.str();
            index += 1;
            //if (bodyBestFrame->bestFrameJpeg.unique()) 
            {
                comm::MicrosecondTimer body_timer;
                body_timer.Start();
                cv::imwrite(bodyJpegName, *(bodyBestFrame->bestFrameJpeg->data));
                body_timer.Stop();
                EASYLOG_INFO << "ptimer body pic cost: " << body_timer.GetMilliseconds();
                EASYLOG_INFO << "save bodyJpegName: " << bodyJpegName;
            }

        }
        /// 最优帧HEAD信息写入文件,至少有1张face or body
        if (bestFrameInfo->alBestFrameJson.length() != 0
            && (!bestFrameInfo->faceBestFrameInfo.empty() || !bestFrameInfo->bodyBestFrameInfo.empty()) 
            )
        {
            ostm.str("");
            ostm << Get_FILES_STORAGE_PATH() << timestamp_str
                << "_"  << bestFrameInfo->alResultUuid
                << "_" << bestFrameInfo->personId
                << "_" << "bestFrameHead.json";
            std::string bestFrameHeadJsonFileName = ostm.str();
            std::ofstream outFile1(bestFrameHeadJsonFileName, std::ios::out | std::ios::binary);
            if (outFile1 && (bestFrameInfo->alBestFrameJson.length() != 0))
            {
                outFile1.write((char*)bestFrameInfo->alBestFrameJson.c_str(), bestFrameInfo->alBestFrameJson.length());
                outFile1.close();
            } else {
                outFile1.close();
                EASYLOG_WARNING << "open file failure error filename: " << bestFrameHeadJsonFileName << " length: " << bestFrameInfo->alBestFrameJson.length();
            }
        }
    }
    total_timer.Stop();
    EASYLOG_INFO << "ptimer total cost: " << total_timer.GetMilliseconds();
    return 0;
}

