
#### 获取maxminddb后缀
#### 暂时只支持x64-windows/x86-windows
function(vcpkg_find_maxminddb_postfix out_POSTFIX)
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(${out_POSTFIX} "d.lib")
    elseif(CMAKE_BUILD_TYPE MATCHES "Release")
        set(${out_POSTFIX} ".lib")
    else()
        set(${out_POSTFIX} ".lib")
    endif()

    set(${out_POSTFIX} ${${out_POSTFIX}} PARENT_SCOPE)
endfunction()

#### 找到maxminddb库的路径
function(vcpkg_find_maxminddb LIB_NAME VCPKG_LIB_DIR out_LIB)
    vcpkg_find_maxminddb_postfix(VCPKG_MMDB_POSTFIX)
    set(LIB_FULLNAME ${LIB_NAME}${VCPKG_MMDB_POSTFIX})

    find_library(${out_LIB} ${LIB_FULLNAME} PATHS ${VCPKG_LIB_DIR})
    if(NOT ${out_LIB})
        message(FATAL_ERROR "Can't find  ${LIB_FULLNAME} ${VCPKG_LIB_DIR}")
    endif()

    set(${out_LIB} ${${out_LIB}} PARENT_SCOPE)
    #message("find ${LIB_FULLNAME} ${out_LIB}")
endfunction()