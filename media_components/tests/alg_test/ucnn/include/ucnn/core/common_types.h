#ifndef UCNN3_COMMON_TYPES_H
#define UCNN3_COMMON_TYPES_H

#include <opencv2/core/types.hpp>

namespace ucnn
{
template <typename T>
struct Box_
{
    float xmin;
    float xmax;
    float ymin;
    float ymax;
    float score;
    int type;

    T extra_info;  // User Specified information
};

template <>
struct Box_<void>
{
    float xmin;
    float xmax;
    float ymin;
    float ymax;
    float score;
    int type;
};

using Box = Box_<void>;
using Boxf = Box_<float>;

using Point = cv::Point;
using Point2f = cv::Point2f;
using Rect = cv::Rect;
using Size = cv::Size;
using Scalar = cv::Scalar;

template <typename T>
using Point_ = cv::Point_<T>;

template <typename T>
using Rect_ = cv::Rect_<T>;

template <typename T>
using Size_ = cv::Size_<T>;

template <typename T>
using Scalar_ = cv::Scalar_<T>;

}  // namespace ucnn

#endif  // UCNN3_COMMON_TYPES_H
