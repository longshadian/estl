set_property(GLOBAL PROPERTY USE_FOLDERS On)

project(media_rtsp CXX)
set(PROJECT_NAME media_rtsp)

#### cmake module路径
set(CMAKE_MODULE_PATH
    ${CMAKE_HOME_DIRECTORY}/cmake
    ${CMAKE_MODULE_PATH}
)

#### cmake模块
include(Functions)

#### 工程文件目录结构
set(SRC_DIR "${CMAKE_HOME_DIRECTORY}/src/media_rtsp")
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
    #### TODO MVSC
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    #### GCC
	set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wconversion -Wshadow -std=c++14 ${CMAKE_CXX_FLAGS}")	
endif()

#### 获取平台目标类型 x86, x64
util_check_platform_size(PLATFORM_TARGET)
message("platform_type: ${PLATFORM_TARGET}")

#### 头文件
include_directories(${CMAKE_HOME_DIRECTORY})
if(VCPKG_DIR)
    #### TODO MSVC
else()
    #### TODO find_package
    include_directories("/home/bolan/works/mylibs/live555/BasicUsageEnvironment/include")
    include_directories("/home/bolan/works/mylibs/live555/groupsock/include")
    include_directories("/home/bolan/works/mylibs/live555/liveMedia/include")
    include_directories("/home/bolan/works/mylibs/live555/UsageEnvironment/include")
endif()

include_directories(${CMAKE_HOME_DIRECTORY}/src/rtmp_components)

#### 生成文件
add_library(${PROJECT_NAME} ${SRC_C_FILES} ${SRC_CPP_FILES})
#set_property(TARGET ${OUTPUT} PROPERTY POSITION_INDEPENDENT_CODE ON)
#set_property(TARGET ${OUTPUT} PROPERTY PREFIX "")

#### 链接选项
if(VCPKG_DIR)
else()
##[[
    target_link_libraries(${PROJECT_NAME} 
        crypto ssl
        pthread
    )
#]]
endif()


