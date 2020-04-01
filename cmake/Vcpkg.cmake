

## 获取 x64 x86
function(check_platform_size out_ARGV)
    if(${CMAKE_SIZEOF_VOID_P} STREQUAL "4")
        set(${out_ARGV} "x86" PARENT_SCOPE)
    else()
        set(${out_ARGV} "x64" PARENT_SCOPE)
    endif()
endfunction()


## 获取
function(check_vcpkg_platform_dir PLATFORM_TARGET VCPKG_DIR     out_PLATFORM_DIR)
    set(VCPKG_INSTALLED_DIR  ${VCPKG_DIR}/installed)
    set(VCPKG_X64WINDOWS_DIR ${VCPKG_INSTALLED_DIR}/x64-windows)
    set(VCPKG_X86WINDOWS_DIR ${VCPKG_INSTALLED_DIR}/x86-windows)

    if(PLATFORM_TARGET STREQUAL "x86")
        set(${out_PLATFORM_DIR} ${VCPKG_X86WINDOWS_DIR})
    else()
        set(${out_PLATFORM_DIR} ${VCPKG_X64WINDOWS_DIR})
    endif()

    set(${out_PLATFORM_DIR} ${${out_PLATFORM_DIR}} PARENT_SCOPE)
endfunction()

## 获取vcpkg 头文件路径
function(check_vcpkg_include_dir PLATFORM_TARGET VCPKG_DIR       out_INCLUDE_DIR)
    check_vcpkg_platform_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_PLATFORM_DIR)
    set(${out_INCLUDE_DIR} ${VCPKG_PLATFORM_DIR}/include)

    set(${out_INCLUDE_DIR} ${${out_INCLUDE_DIR}} PARENT_SCOPE)
endfunction()

## 获取vcpkg lib路径，需要指定 -DCMAKE_BUILD_TYPE=Release|Debug
function(check_vcpkg_lib_dir PLATFORM_TARGET VCPKG_DIR       out_LIB_DIR)
    check_vcpkg_platform_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_PLATFORM_DIR)

    if(CMAKE_BUILD_TYPE MATCHES "Release")
        set(${out_LIB_DIR} ${VCPKG_PLATFORM_DIR}/lib)
    elseif(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(${out_LIB_DIR} "${VCPKG_PLATFORM_DIR}/debug/lib")
    elseif(CMAKE_BUILD_TYPE MATCHES "MinSizeRel")
        set(${out_LIB_DIR} "")
    elseif(CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
        set(${out_LIB_DIR} "")
    else()
        set(${out_LIB_DIR} "")
    endif()

    set(${out_LIB_DIR} ${${out_LIB_DIR}} PARENT_SCOPE)
endfunction()
