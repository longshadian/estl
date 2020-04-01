cmake_minimum_required(VERSION 3.0)
project(Convert)

if(USE_CMAKE)
    find_package(Boost REQUIRED)
    include_directories(${Boost_INCLUDE_DIRS})    
elseif(USE_VCPKG)
    #### cmake模块
    set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake" ${CMAKE_MODULE_PATH})
    include(Vcpkg)
    include(VcpkgBoost)

    #### vcpkg 目录
    set(VCPKG_DIR "c:/vcpkg")

    #### 获取平台目标类型 x86, x64
    check_platform_size(PLATFORM_TARGET)
    ## message("platform_type: ${PLATFORM_TARGET}")
    check_vcpkg_include_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_INC_DIR)
    #message("VCPKG_INC_DIR: ${VCPKG_INC_DIR}")
    include_directories(${VCPKG_INC_DIR})
endif()

add_executable(Convert Convert.cpp)
if(USE_CMAKE)
    
elseif(USE_VCPKG)
    check_vcpkg_lib_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_LIB_DIR)
    check_vcpkg_boost_lib("boost_system" ${VCPKG_LIB_DIR} BOOST_SYSTEM)

    set(Boost_LIBRARIES ${BOOST_SYSTEM})
endif()
target_link_libraries(Convert PRIVATE ${Boost_LIBRARIES})