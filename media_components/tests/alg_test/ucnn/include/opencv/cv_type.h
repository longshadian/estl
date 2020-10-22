#ifndef OPENCV_UCNN_CORE_TYPE_H
#define OPENCV_UCNN_CORE_TYPE_H

#ifdef HAVE_OPENCV
#include <opencv2/core/cvdef.h>
#else
#include <algorithm>
#include <limits>
#define CV_CN_MAX 512
#define CV_CN_SHIFT 3
#define CV_DEPTH_MAX (1 << CV_CN_SHIFT)

#define CV_8U 0
#define CV_8S 1
#define CV_16U 2
#define CV_16S 3
#define CV_32S 4
#define CV_32F 5
#define CV_64F 6
#define CV_16F 7

#define CV_MAT_DEPTH_MASK (CV_DEPTH_MAX - 1)
#define CV_MAT_DEPTH(flags) ((flags)&CV_MAT_DEPTH_MASK)

#define CV_MAKETYPE(depth, cn) (CV_MAT_DEPTH(depth) + (((cn)-1) << CV_CN_SHIFT))
#define CV_MAKE_TYPE CV_MAKETYPE

#define CV_8UC1 CV_MAKETYPE(CV_8U, 1)
#define CV_8UC2 CV_MAKETYPE(CV_8U, 2)
#define CV_8UC3 CV_MAKETYPE(CV_8U, 3)
#define CV_8UC4 CV_MAKETYPE(CV_8U, 4)
#define CV_8UC(n) CV_MAKETYPE(CV_8U, (n))

#define CV_8SC1 CV_MAKETYPE(CV_8S, 1)
#define CV_8SC2 CV_MAKETYPE(CV_8S, 2)
#define CV_8SC3 CV_MAKETYPE(CV_8S, 3)
#define CV_8SC4 CV_MAKETYPE(CV_8S, 4)
#define CV_8SC(n) CV_MAKETYPE(CV_8S, (n))

#define CV_16UC1 CV_MAKETYPE(CV_16U, 1)
#define CV_16UC2 CV_MAKETYPE(CV_16U, 2)
#define CV_16UC3 CV_MAKETYPE(CV_16U, 3)
#define CV_16UC4 CV_MAKETYPE(CV_16U, 4)
#define CV_16UC(n) CV_MAKETYPE(CV_16U, (n))

#define CV_16SC1 CV_MAKETYPE(CV_16S, 1)
#define CV_16SC2 CV_MAKETYPE(CV_16S, 2)
#define CV_16SC3 CV_MAKETYPE(CV_16S, 3)
#define CV_16SC4 CV_MAKETYPE(CV_16S, 4)
#define CV_16SC(n) CV_MAKETYPE(CV_16S, (n))

#define CV_32SC1 CV_MAKETYPE(CV_32S, 1)
#define CV_32SC2 CV_MAKETYPE(CV_32S, 2)
#define CV_32SC3 CV_MAKETYPE(CV_32S, 3)
#define CV_32SC4 CV_MAKETYPE(CV_32S, 4)
#define CV_32SC(n) CV_MAKETYPE(CV_32S, (n))

#define CV_32FC1 CV_MAKETYPE(CV_32F, 1)
#define CV_32FC2 CV_MAKETYPE(CV_32F, 2)
#define CV_32FC3 CV_MAKETYPE(CV_32F, 3)
#define CV_32FC4 CV_MAKETYPE(CV_32F, 4)
#define CV_32FC(n) CV_MAKETYPE(CV_32F, (n))

#define CV_64FC1 CV_MAKETYPE(CV_64F, 1)
#define CV_64FC2 CV_MAKETYPE(CV_64F, 2)
#define CV_64FC3 CV_MAKETYPE(CV_64F, 3)
#define CV_64FC4 CV_MAKETYPE(CV_64F, 4)
#define CV_64FC(n) CV_MAKETYPE(CV_64F, (n))

#define CV_16FC1 CV_MAKETYPE(CV_16F, 1)
#define CV_16FC2 CV_MAKETYPE(CV_16F, 2)
#define CV_16FC3 CV_MAKETYPE(CV_16F, 3)
#define CV_16FC4 CV_MAKETYPE(CV_16F, 4)
#define CV_16FC(n) CV_MAKETYPE(CV_16F, (n))

/****************************************************************************************\
*                                  Matrix type (Mat)                                     *
\****************************************************************************************/

#define CV_MAT_CN_MASK ((CV_CN_MAX - 1) << CV_CN_SHIFT)
#define CV_MAT_CN(flags) ((((flags)&CV_MAT_CN_MASK) >> CV_CN_SHIFT) + 1)

/** Size of each channel item,
   0x28442211 = 0010 1000 0100 0100 0010 0010 0001 0001 ~ array of sizeof(arr_type_elem) */
#define CV_ELEM_SIZE1(type) ((0x28442211 >> CV_MAT_DEPTH(type) * 4) & 15)

#define CV_ELEM_SIZE(type) (CV_MAT_CN(type) * CV_ELEM_SIZE1(type))

#endif

#define CV_MAX (CV_MAKETYPE(7, (CV_CN_MAX)) + 1)

// below are ive types

#define IVE_U8C1 (CV_MAX)
#define IVE_S8C1 (CV_MAX + 1)

#define IVE_YUV420SP (CV_MAX + 2)
#define IVE_YUV422SP (CV_MAX + 3)
#define IVE_YUV420P (CV_MAX + 4)
#define IVE_YUV422P (CV_MAX + 5)

#define IVE_S8C2_PACKAGE (CV_MAX + 6)
#define IVE_S8C2_PLANAR (CV_MAX + 7)

#define IVE_S16C1 (CV_MAX + 8)
#define IVE_U16C1 (CV_MAX + 9)

#define IVE_U8C3_PACKAGE (CV_MAX + 10)
#define IVE_U8C3_PLANAR (CV_MAX + 11)

#define IVE_S32C1 (CV_MAX + 12)
#define IVE_U32C1 (CV_MAX + 13)

#define IVE_S64C1 (CV_MAX + 14)
#define IVE_U64C1 (CV_MAX + 15)

#endif