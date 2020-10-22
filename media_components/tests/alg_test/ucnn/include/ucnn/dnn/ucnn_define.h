//
//  MNNDefine.h
//  MNN
//
//  Created by MNN on 2018/08/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef UCNNDefine_h
#define UCNNDefine_h

#include <assert.h>
#include <stdio.h>

#if defined(__APPLE__)
#include "TargetConditionals.h"
#if TARGET_OS_IPHONE
#define UCNN_BUILD_FOR_IOS
#endif
#endif

#ifdef UCNN_USE_LOGCAT
#include <android/log.h>
#define UCNN_ERROR(format, ...) __android_log_print(ANDROID_LOG_ERROR, "UCNNJNI", format, ##__VA_ARGS__)
#define UCNN_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "UCNNJNI", format, ##__VA_ARGS__)
#else
#define UCNN_PRINT(format, ...) printf(format, ##__VA_ARGS__)
#define UCNN_ERROR(format, ...) printf(format, ##__VA_ARGS__)
#endif

#ifdef DEBUG
#define UCNN_ASSERT(x)                                            \
    {                                                             \
        int res = (x);                                            \
        if (!res)                                                 \
        {                                                         \
            UCNN_ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
            assert(res);                                          \
        }                                                         \
    }
#else
#define UCNN_ASSERT(x)                                            \
    {                                                             \
        int res = (x);                                            \
        if (!res)                                                 \
        {                                                         \
            UCNN_ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
        }                                                         \
    }
#endif

#if defined(_MSC_VER)
#if defined(BUILDING_UCNN_DLL)
#define UCNN_PUBLIC __declspec(dllexport)
#elif defined(USING_UCNN_DLL)
#define UCNN_PUBLIC __declspec(dllimport)
#else
#define UCNN_PUBLIC
#endif
#else
#define UCNN_PUBLIC __attribute__((visibility("default")))
#endif

#endif /* UCNNDefine_h */
