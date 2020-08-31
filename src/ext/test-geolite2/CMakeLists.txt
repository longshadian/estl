cmake_minimum_required(VERSION 3.0)
project(TestMaxminddb)

include_directories(${CMAKE_HOME_DIRECTORY})

if(USE_CMAKE)
    #find_package(nlohmann_json CONFIG REQUIRED)
    #include_directories(${Boost_INCLUDE_DIRS})    
elseif(VCPKG_DIR)
    #### cmake模块
    set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake" ${CMAKE_MODULE_PATH})
    include(Functions)
    include(Vcpkg)
    include(VcpkgMaxminddb)

    #### vcpkg 目录
    #message("VCPKG_DIR: ${VCPKG_DIR}")

    #### 获取平台目标类型 x86, x64
    util_check_platform_size(PLATFORM_TARGET)
    vcpkg_find_include_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_INC_DIR)
    include_directories(${VCPKG_INC_DIR})
endif()

add_executable(Maxminddb Main.cpp)
if(USE_CMAKE)
    #target_link_libraries(Maxminddb PRIVATE nlohmann_json nlohmann_json::nlohmann_json)
elseif(VCPKG_DIR)
    vcpkg_find_lib_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_LIB_DIR)
    vcpkg_find_maxminddb("libmaxminddb" ${VCPKG_LIB_DIR} MAXMINDDB)
    target_link_libraries(Maxminddb PRIVATE ${MAXMINDDB}  ws2_32 Rpcrt4 shlwapi)
endif()
