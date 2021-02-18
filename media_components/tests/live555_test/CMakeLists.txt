set_property(GLOBAL PROPERTY USE_FOLDERS On)

set(PROJECT_NAME live555_test)
project(${PROJECT_NAME} CXX)

#### cmake module路径
set(CMAKE_MODULE_PATH
    "${CMAKE_HOME_DIRECTORY}/cmake"
    "${CMAKE_MODULE_PATH}"
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

#### 编译选项
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
	set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wconversion -Wshadow -std=c++14 ${CMAKE_CXX_FLAGS}")	
endif()

#### 获取平台目标类型 x86, x64
util_check_platform_size(PLATFORM_TARGET)
message("platform_type: ${PLATFORM_TARGET}")

include_directories(${CMAKE_HOME_DIRECTORY}/src/media_rtsp)
include_directories("/home/bolan/works/mylibs/live555/BasicUsageEnvironment/include")
include_directories("/home/bolan/works/mylibs/live555/groupsock/include")
include_directories("/home/bolan/works/mylibs/live555/liveMedia/include")
include_directories("/home/bolan/works/mylibs/live555/UsageEnvironment/include")

link_directories("/home/bolan/works/mylibs/live555/lib")

#### 生成可执行文件
link_directories("/home/bolan/libs/lib")
add_executable(${PROJECT_NAME} 
    ${SRC_C_FILES}
    ${SRC_CPP_FILES}
)

#### 链接选项
target_link_libraries(${PROJECT_NAME} 
    media_rtsp
    libliveMedia.a libgroupsock.a libBasicUsageEnvironment.a libUsageEnvironment.a
    crypto ssl
    pthread
)

