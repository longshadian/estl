cmake_minimum_required(VERSION 3.0)
project(PluginBoost)


SET(OUTPUT_DIR ${CMAKE_BINARY_DIR}/output)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}/debug/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}/debug/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}/debug/bin)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}/release/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}/release/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}/release/lib)

if(USE_CMAKE)
    find_package(Boost COMPONENTS filesystem system REQUIRED)
    include_directories(${Boost_INCLUDE_DIRS})    
elseif(USE_VCPKG)
    #### cmake模块
    set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake" ${CMAKE_MODULE_PATH})
    include(Vcpkg)
    include(VcpkgBoost)
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

#INCLUDE_DIRECTORIES(${CMAKE_HOME_DIRECTORY})

add_library(PluginBoost SHARED PluginBoost.cpp)
if(USE_CMAKE)

elseif(USE_VCPKG)
    vcpkg_find_lib_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_LIB_DIR)
    vcpkg_find_boost("boost_filesystem" ${VCPKG_LIB_DIR} BOOST_FILESYSTEM)
    vcpkg_find_boost("boost_system" ${VCPKG_LIB_DIR} BOOST_SYSTEM)

    set(Boost_LIBRARIES ${BOOST_FILESYSTEM} ${BOOST_SYSTEM})
endif()
target_link_libraries(PluginBoost PRIVATE ${Boost_LIBRARIES})