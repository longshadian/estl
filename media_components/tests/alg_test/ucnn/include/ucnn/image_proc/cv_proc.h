//
// Created by pengguanhai on 19-11-21.
//

#ifndef UCNN_CV_PROC_H
#define UCNN_CV_PROC_H
#include <memory>
#include <vector>

#include "ucnn/core/common_types.h"
#include "ucnn/core/mat.h"
namespace ucnn
{
class Mat;

enum ImageReadModes
{
    IMREAD_UNCHANGED = -1,
    IMREAD_GRAYSCALE = 0,
    IMREAD_COLOR = 1,
    IMREAD_ANYDEPTH = 2,
    IMREAD_ANYCOLOR = 4,
    IMREAD_LOAD_GDAL = 8,
    IMREAD_REDUCED_GRAYSCALE_2 = 16,
    IMREAD_REDUCED_COLOR_2 = 17,
    IMREAD_REDUCED_GRAYSCALE_4 = 32,
    IMREAD_REDUCED_COLOR_4 = 33,
    IMREAD_REDUCED_GRAYSCALE_8 = 64,
    IMREAD_REDUCED_COLOR_8 = 65,
    IMREAD_IGNORE_ORIENTATION = 128
};

Mat ReadImage(const std::string &image_path, int flags = IMREAD_COLOR);

int WriteImage(const Mat &mat, const std::string &image_path);

/// \brief 从原图src裁剪roiRect区域crop到目标图像dst,裁剪区域左上角点与目标图像的globalPoint重合
int CropAndPad(const Mat &src, Mat &dst, const Rect &roi_rect = {0, 0, 0, 0}, const Point &global_point = {0, 0});

/// \brief 图像数据备份
int DeepClone(const Mat &src, Mat &dst);

/// \brief 从src图像裁剪出roiRect区域图像存放至dst
int RoiCrop(const Mat &src, Mat &dst, const Rect &roi_rect = {0, 0, 0, 0});

/// \brief 把原图像copy至目标图像，原图像左上角点与目标图像globalPoint重合
int PadImage(const Mat &src, Mat &dst, const Point &global_point = {0, 0}, const Scalar &value = Scalar());

/// \brief 图像类型转换
int CvtColor(const Mat &src, Mat &dst);

/// \brief 把src图像缩放至dst图像大小
int Resize(const Mat &src, Mat &dst);

int Resize(const Mat &src, Mat &dst, Size size);

/// \brief 从src图像裁剪多个roiRectMulti区域，统一缩放并存储至dstMulti
int Resize(const Mat &src, const std::vector<Rect> &roi_rect_multi, std::vector<Mat> &dst_multi);

/// \brief 根据图像尺寸imgSize和图像拼接数量mergeNum,获取拼接图像行数mergeBlockRows和列数mergeBlockCols
int GetMergeWay(const Size &img_size, const int merge_num, int &merge_block_rows, int &merge_block_cols);

/// \brief 根据各块图像srcMulti的编号位置indexMulti做拼接存放至目标区域dst
int MergeReform(const std::vector<Mat> &src_multi, const std::vector<int> &index_multi, Mat &dst);

/// \brief 对源图像src根据变换矩阵rotationMat做仿射变换得到目标图像dst
int WarpAffine(const Mat &src, const float rotation_mat[6], Mat &dst);

void FillPoly(Mat &img, const Point **pts, const int *npts, int ncontours, const Scalar &color);
}  // namespace ucnn

#endif  // UCNN_CV_PROC_H
