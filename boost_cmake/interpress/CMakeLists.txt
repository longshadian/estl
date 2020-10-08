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
    include(Functions)

    #### vcpkg 目录
    set(VCPKG_DIR "c:/vcpkg")

    #### 获取平台目标类型 x86, x64
    util_check_platform_size(PLATFORM_TARGET)
    ## message("platform_type: ${PLATFORM_TARGET}")
    vcpkg_find_include_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_INC_DIR)
    #message("VCPKG_INC_DIR: ${VCPKG_INC_DIR}")
    include_directories(${VCPKG_INC_DIR})
endif()

add_executable(Interpress Interpress.cpp)
if(USE_CMAKE)

elseif(USE_VCPKG)
    vcpkg_find_lib_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_LIB_DIR)
    vcpkg_find_boost("boost_system" ${VCPKG_LIB_DIR} BOOST_SYSTEM)
    vcpkg_find_zlib("zlib" ${VCPKG_LIB_DIR} ZLIB)

    set(Boost_LIBRARIES  ${ZLIB})
endif()
target_link_libraries(Interpress PRIVATE ${Boost_LIBRARIES})