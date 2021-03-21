cmake_minimum_required(VERSION 3.0)
set_property(GLOBAL PROPERTY USE_FOLDERS On)

project(ffmpeg CXX)
set(PROJECT_NAME ffmpeg)

#### cmake module路径
set(CMAKE_MODULE_PATH
    "${CMAKE_HOME_DIRECTORY}/cmake"
    ${CMAKE_MODULE_PATH}
    )

#### cmake模块
include(Functions)
include(Vcpkg)
include(VcpkgFFmpeg)

#### 工程文件目录结构
set(SRC_DIR "${CMAKE_HOME_DIRECTORY}/src")
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
    add_definitions("/std:c++14")
    add_definitions("/D _WIN32_WINNT=0x0601")
    add_definitions("/D _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS")
    add_definitions("/D _CRT_SECURE_NO_WARNINGS")
    add_definitions("/D SRS_AUTO_SSL")
    add_definitions("/utf-8")     ### utf-8 no-bom warning C4819: 
    
    #### ffmpeg编译错误
    add_definitions("/sdl-")      ### warning C4996: 'avcodec_encode_video2': 被声明为已否决
    add_definitions("/D__STDC_CONSTANT_MACROS")   ### \libavutil\common.h(30,1): fatal error C1189: #error:  missing -D__STDC_CONSTANT_MACROS / #define __STDC_CONSTANT_MACROS

    #### live555 编译后出错
    #add_definitions("/NODEFAULTLIB:library")   #### warning LNK4098: 默认库“LIBCMTD”与其他库的使用冲突；请使用 /NODEFAULTLIB:library
    add_definitions("/MTd")     #### vcpkg只有live555的静态库
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
	set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wconversion -Wshadow -std=c++17 -Wl,-Bsymbolic ${CMAKE_CXX_FLAGS}")	
endif()

#### 获取平台目标类型 x86, x64
util_check_platform_size(PLATFORM_TARGET)
message("platform_type: ${PLATFORM_TARGET}")

#### 头文件
include_directories(${CMAKE_HOME_DIRECTORY})
if(VCPKG_DIR)
    vcpkg_find_include_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_INC_DIR)
    message("VCPKG_INC_DIR: ${VCPKG_INC_DIR}")
    include_directories(${VCPKG_INC_DIR})

    #### TODO 动态判断头文件
    include_directories("D:/vcpkg/installed/x64-windows-static/include")
endif()

####
include_directories(${CMAKE_HOME_DIRECTORY}/src)

#### 生成可执行文件
add_executable(${PROJECT_NAME} 
    ${SRC_C_FILES}
    ${SRC_CPP_FILES}
    ${SRC_H_FILES}
    ${SRC_HPP_FILES}
)

#### 链接选项
if(VCPKG_DIR)
    vcpkg_find_lib_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_LIB_DIR)

    set(VCPKG_STATIC_LIB_DIR "D:/vcpkg/installed/x64-windows-static/debug/lib")
    
    #### openssl
    #check_vcpkg_ssl_lib("libssl" ${VCPKG_LIB_DIR} SSL)
    #check_vcpkg_ssl_lib("libcrypto" ${VCPKG_LIB_DIR} CRYPTO)

    #### ffmpeg
    vcpkg_find_ffmpeg("avcodec" ${VCPKG_LIB_DIR} lib_avcodec)
    vcpkg_find_ffmpeg("avformat" ${VCPKG_LIB_DIR} lib_avformat)
    vcpkg_find_ffmpeg("avutil" ${VCPKG_LIB_DIR} lib_avutil)
    vcpkg_find_ffmpeg("avfilter" ${VCPKG_LIB_DIR} lib_avfilter)
    vcpkg_find_ffmpeg("swscale" ${VCPKG_LIB_DIR} lib_swscale)
    vcpkg_find_ffmpeg("swresample" ${VCPKG_LIB_DIR} lib_swresample)

    #vcpkg_find_lib_live555("liveMedia" ${VCPKG_STATIC_LIB_DIR} lib_liveMedia)
    #vcpkg_find_lib_live555("groupsock" ${VCPKG_STATIC_LIB_DIR} lib_groupsock)
    #vcpkg_find_lib_live555("BasicUsageEnvironment" ${VCPKG_STATIC_LIB_DIR} lib_BasicUsageEnvironment)
    #vcpkg_find_lib_live555("UsageEnvironment" ${VCPKG_STATIC_LIB_DIR} lib_UsageEnvironment)
       
    target_link_libraries(${PROJECT_NAME} 
        #### ffmpeg
        ${lib_avcodec}  ${lib_avformat} ${lib_avutil} ${lib_avfilter} ${lib_swscale} ${lib_swresample}

        #### live555
        #${lib_liveMedia} ${lib_groupsock} ${lib_BasicUsageEnvironment} ${lib_UsageEnvironment}

        #${SSL} ${CRYPTO}
        WS2_32
    )
else()
    target_link_libraries(${PROJECT_NAME} 
        pthread
    )
endif()


