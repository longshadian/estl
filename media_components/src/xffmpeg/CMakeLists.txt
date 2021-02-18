set_property(GLOBAL PROPERTY USE_FOLDERS On)

project(xffmpeg CXX)
set(PROJECT_NAME xffmpeg)

#### cmake module路径
set(CMAKE_MODULE_PATH
    ${CMAKE_HOME_DIRECTORY}/cmake
    ${CMAKE_MODULE_PATH}
)

#### cmake模块
include(Functions)

#### 工程文件目录结构
set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}")
util_auto_sources(SRC_C_FILES "*.c" "RECURSE" "${SRC_DIR}")
util_auto_sources(SRC_CPP_FILES "*.cpp" "RECURSE" "${SRC_DIR}")
util_auto_sources(SRC_H_FILES "*.h" "RECURSE" "${SRC_DIR}")
util_auto_sources(SRC_HPP_FILES "*.hpp" "RECURSE" "${SRC_DIR}")
util_auto_source_group(src ${SRC_DIR} 
    ${SRC_C_FILES}
    ${SRC_CPP_FILES}
    ${SRC_H_FILES}
    ${SRC_HPP_FILES}
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS false)
set(CMAKE_CXX_STANDARD_REQUIRED true)

#### 编译选项
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    #### TODO MVSC
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    #### GCC
    set(CMAKE_CXX_FLAGS "-Wall ${CMAKE_CXX_FLAGS}")	
endif()

#### 头文件
include_directories(${CMAKE_HOME_DIRECTORY}/src)

#### 生成文件
add_library(${PROJECT_NAME} ${SRC_C_FILES} ${SRC_CPP_FILES})

#### 链接选项
target_link_libraries(${PROJECT_NAME} 
    avformat avcodec avutil   avfilter avdevice  postproc swresample swscale  x264 x265
    crypto ssl
    pthread
)


