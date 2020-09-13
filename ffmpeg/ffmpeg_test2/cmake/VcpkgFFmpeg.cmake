
## 获取vcpkg ffmpeg库后缀
## 暂时只支持x64-windows/x86-windows
function(check_vcpkg_ffmpeg_postfix out_POSTFIX)
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(${out_POSTFIX} ".lib")
    elseif(CMAKE_BUILD_TYPE MATCHES "Release")
        set(${out_POSTFIX} ".lib")
    else()
        set(${out_POSTFIX} ".lib")
    endif()

    set(${out_POSTFIX} ${${out_POSTFIX}} PARENT_SCOPE)
endfunction()

## 找到ffmpeg库的路径
## set(VCPKG_LIB_DIR "c:/vcpkg/installed/x64-windows")
## check_vcpkg_ffmpeg_lib("avcodec" ${VCPKG_LIB_DIR} lib_avcodec)
## check_vcpkg_ffmpeg_lib("avformat" ${VCPKG_LIB_DIR} lib_avformat)
## check_vcpkg_ffmpeg_lib("avutil" ${VCPKG_LIB_DIR} lib_avutil)
## check_vcpkg_ffmpeg_lib("avfilter" ${VCPKG_LIB_DIR} lib_avfilter)
## check_vcpkg_ffmpeg_lib("swscale" ${VCPKG_LIB_DIR} lib_swscale)
## check_vcpkg_ffmpeg_lib("swresample" ${VCPKG_LIB_DIR} lib_swresample)
function(check_vcpkg_ffmpeg_lib LIB_NAME VCPKG_LIB_DIR out_LIB)
    check_vcpkg_ffmpeg_postfix (VCPKG_LIB_POSTFIX)
    set(LIB_FULLNAME ${LIB_NAME}${VCPKG_LIB_POSTFIX})

    find_library(${out_LIB} ${LIB_FULLNAME} PATHS ${VCPKG_LIB_DIR})
    if(NOT ${out_LIB})
        message(FATAL_ERROR "Can't find  ${LIB_FULLNAME} ${VCPKG_LIB_DIR}")
    endif()

    set(${out_LIB} ${${out_LIB}} PARENT_SCOPE)
    #message("find ${LIB_FULLNAME} ${out_LIB}")
endfunction()