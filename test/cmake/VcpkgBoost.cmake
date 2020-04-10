
## 获取vcpkg boost库后缀，例如 boost_fiber-vc140-mt-gd.lib boost_filesystem-vc140-mt.lib
## 暂时只支持x64-windows/x86-windows
function(check_vcpkg_boost_postfix out_POSTFIX)
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(${out_POSTFIX} "-vc140-mt-gd.lib")
    elseif(CMAKE_BUILD_TYPE MATCHES "Release")
        set(${out_POSTFIX} "-vc140-mt.lib")
    else()
        set(${out_POSTFIX} "-vc140.lib")
    endif()

    set(${out_POSTFIX} ${${out_POSTFIX}} PARENT_SCOPE)
endfunction()

## 找到boost库的路径
## set(VCPKG_LIB_DIR "c:/vcpkg/installed/x64-windows")
## check_vcpkg_boost_lib("boost_system" ${VCPKG_LIB_DIR} LIB_BOOST_SYSTEM)
## message("boost_system: ${LIB_BOOST_SYSTEM}")
function(check_vcpkg_boost_lib BOOST_LIB_NAME VCPKG_LIB_DIR out_LIB)
    check_vcpkg_boost_postfix(VCPKG_BOOST_POSTFIX)
    set(LIB_FULLNAME ${BOOST_LIB_NAME}${VCPKG_BOOST_POSTFIX})

    find_library(${out_LIB} ${LIB_FULLNAME} PATHS ${VCPKG_LIB_DIR})
    if(NOT ${out_LIB})
        message(FATAL_ERROR "Can't find  ${LIB_FULLNAME} ${VCPKG_LIB_DIR}")
    endif()

    set(${out_LIB} ${${out_LIB}} PARENT_SCOPE)
    #message("find ${LIB_FULLNAME} ${out_LIB}")
endfunction()