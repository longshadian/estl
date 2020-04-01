cmake_minimum_required(VERSION 3.0)
project(Interpress)

if(USE_CMAKE)
    find_package(Boost REQUIRED)
    include_directories(${Boost_INCLUDE_DIRS})    
elseif(USE_VCPKG)
    #### cmake模块
    set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake" ${CMAKE_MODULE_PATH})
    include(Vcpkg)
    include(VcpkgBoost)
    include(VcpkgZlib)

    #### vcpkg 目录
    set(VCPKG_DIR "c:/vcpkg")

    #### 获取平台目标类型 x86, x64
    check_platform_size(PLATFORM_TARGET)
    ## message("platform_type: ${PLATFORM_TARGET}")
    check_vcpkg_include_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_INC_DIR)
    #message("VCPKG_INC_DIR: ${VCPKG_INC_DIR}")
    include_directories(${VCPKG_INC_DIR})
endif()

add_executable(Interpress Interpress.cpp)
if(USE_CMAKE)

elseif(USE_VCPKG)
    check_vcpkg_lib_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_LIB_DIR)
    check_vcpkg_boost_lib("boost_system" ${VCPKG_LIB_DIR} BOOST_SYSTEM)
    check_vcpkg_zlib_lib("zlib" ${VCPKG_LIB_DIR} ZLIB)

    set(Boost_LIBRARIES  ${ZLIB})
endif()
target_link_libraries(Interpress PRIVATE ${Boost_LIBRARIES})