cmake_minimum_required(VERSION 3.18)
set(CMAKE_PROJECT_NAME "ffmpeg_test")
project(${CMAKE_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS false)
set(CMAKE_CXX_STANDARD_REQUIRED true)

#include("/home/bolan/works/uniubi/algorithm_platform/build/conanbuildinfo.cmake")
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()


add_executable(fftest main.cpp xffmpeg.cpp)

target_link_libraries(fftest  avformat avcodec avutil   avfilter avdevice  postproc swresample swscale  x264 x265 pthread z bz2 lzma)