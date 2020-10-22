//
// Created by pengguanhai on 2020/7/9.
//

#ifndef AIRPORT_TRACKING_HUMAN_SNAP_PUBLIC_H
#define AIRPORT_TRACKING_HUMAN_SNAP_PUBLIC_H

#include <memory>
#include <vector>

#include "common/airport_common.h"
#include "ucnn/core/common_types.h"

namespace ucnn
{
class PersonSnap;

/// 行人抓拍类
class PersonSnapPublic
{
  public:
    /// 行人抓拍类构造函数
    /// \param model_prefix 输入模型路径
    /// \param roi_multi 输入多个感兴趣区域
    /// \param verify_name 模型校验证书路径
    /// \param eth_name 校验网口地址
    PersonSnapPublic(const std::string &model_prefix, const std::vector<Rect> &roi_multi, std::string verify_name,
                     std::string eth_name = "enp2s0", int device_id = 0);

    /// 行人抓拍类析构函数
    ~PersonSnapPublic();

    /// 行人抓拍函数
    /// \param frame_info_multi 输入多个图像数据
    /// \param config_param 输入抓拍配置参数
    /// \param success_id_multi 输入已经匹配上的ID号(不再上传)
    /// \param bestFrame_info_multi 输出最优帧数据
    /// \return 返回实时行人信息
    std::vector<std::vector<uniHumanInfo>> snap(const std::vector<uniFrameInfoAlg> &frame_info_multi,
                                                uniConfigParam config_param,
                                                std::vector<std::vector<std::string>> &success_id_multi,
                                                std::vector<std::vector<uniBestFrameInfo>> &bestFrame_info_multi);

  private:
    std::shared_ptr<PersonSnap> person_snap_ptr_;
};

}  // namespace ucnn

#endif  // AIRPORT_TRACKING_HUMAN_SNAP_PUBLIC_H
