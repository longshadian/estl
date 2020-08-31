cmake_minimum_required(VERSION 3.0)
set_property(GLOBAL PROPERTY USE_FOLDERS On)

project(nvcodec)
set(PROJECT_NAME nvcodec)

set(CMAKE_MODULE_PATH
    "${CMAKE_HOME_DIRECTORY}/cmake"
    ${CMAKE_MODULE_PATH}
    )

#### cmake模块
include(Functions)

#### 工程文件目录结构
set(SRC_DIR "${CMAKE_HOME_DIRECTORY}/nvcodec")
auto_sources(SRC_C_FILES "*.c" "RECURSE" "${SRC_DIR}")
auto_sources(SRC_CPP_FILES "*.cpp" "RECURSE" "${SRC_DIR}")
auto_sources(SRC_H_FILES "*.h" "RECURSE" "${SRC_DIR}")
auto_source_group(src ${SRC_DIR} ${SRC_C_FILES} ${SRC_CPP_FILES} ${SRC_HFILES})

#### 编译选项
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_definitions(/std:c++17)
    add_definitions("/D _WIN32_WINNT=0x0601")
    add_definitions("/D _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS")
    add_definitions("/D _CRT_SECURE_NO_WARNINGS")
    add_definitions("/utf-8")     ### utf-8 no-bom warning C4819: 
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
	#set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wconversion -Wshadow -std=c++17 ${CMAKE_CXX_FLAGS}")	
	set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")	
endif()

#### 获取平台目标类型 x86, x64
#### 头文件
include_directories(
    "${CMAKE_HOME_DIRECTORY}/nvcodec"
    )

include_directories(
    "/usr/local/cuda-10.2/include"
    #"/usr/local/cuda-10.1/include"
    "/home/bolan/works/libs/ffmpeg-4.3.1"

    "/home/bolan/works/libs/Video_Codec_SDK_10.0.26"
    #"/home/bolan/works/libs/Video_Codec_SDK_9.1.23"
)

link_directories(
    #"/usr/local/cuda-10.1/lib64"
    "/usr/local/cuda-10.2/lib64"
    "/home/bolan/works/libs/Video_Codec_SDK_10.0.26/Lib/linux/stubs/x86_64"
    #"/home/bolan/works/libs/Video_Codec_SDK_9.1.23/Lib/linux/stubs/x86_64"

    "/home/bolan/works/libs/ffmpeg-4.3.1/libavdevice"
    "/home/bolan/works/libs/ffmpeg-4.3.1/libavformat"
    "/home/bolan/works/libs/ffmpeg-4.3.1/libavfilter"
    "/home/bolan/works/libs/ffmpeg-4.3.1/libavcodec"
    "/home/bolan/works/libs/ffmpeg-4.3.1/libswresample"
    "/home/bolan/works/libs/ffmpeg-4.3.1/libswscale"
    "/home/bolan/works/libs/ffmpeg-4.3.1/libavutil"

)

#### 编译成动态库
#add_library(${PROJECT_NAME} SHARED ${SRC_FILES} ${SRC_HFILES})
#set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
#set_property(TARGET ${PROJECT_NAME} PROPERTY PREFIX "")
#target_link_libraries(${PROJECT_NAME} pthread)

add_executable(${PROJECT_NAME} ${SRC_C_FILES} ${SRC_CPP_FILES} ${SRC_H_FILES})
target_link_libraries(${PROJECT_NAME} 
    #### "liveMedia" "groupsock" "BasicUsageEnvironment" "UsageEnvironment" 
    "avdevice" "avformat" "avfilter" "avcodec" "swresample" "swscale" "avutil"
    "nvcuvid" "cuda"
    "ssl" "crypto" 
    "lzma" "m" "z" "bz2"
    "pthread"
)

