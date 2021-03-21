

## 获取
function(vcpkg_find_platform_dir PLATFORM_TARGET VCPKG_DIR     out_PLATFORM_DIR)
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
function(vcpkg_find_include_dir PLATFORM_TARGET VCPKG_DIR       out_INCLUDE_DIR)
    vcpkg_find_platform_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_PLATFORM_DIR)
    set(${out_INCLUDE_DIR} ${VCPKG_PLATFORM_DIR}/include)

    set(${out_INCLUDE_DIR} ${${out_INCLUDE_DIR}} PARENT_SCOPE)
endfunction()

## 获取vcpkg lib路径，需要指定 -DCMAKE_BUILD_TYPE=Release|Debug
function(vcpkg_find_lib_dir PLATFORM_TARGET VCPKG_DIR       out_LIB_DIR)
    vcpkg_find_platform_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_PLATFORM_DIR)

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
