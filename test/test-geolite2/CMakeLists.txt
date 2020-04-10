cmake_minimum_required(VERSION 3.0)
project(TestMaxminddb)

include_directories(${CMAKE_HOME_DIRECTORY})

if(USE_CMAKE)
    #find_package(nlohmann_json CONFIG REQUIRED)
    #include_directories(${Boost_INCLUDE_DIRS})    
elseif(VCPKG_DIR)
    #### cmake模块
    set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake" ${CMAKE_MODULE_PATH})
    include(Vcpkg)
    include(VcpkgMaxminddb)

    #### vcpkg 目录
    #message("VCPKG_DIR: ${VCPKG_DIR}")

    #### 获取平台目标类型 x86, x64
    check_platform_size(PLATFORM_TARGET)
    check_vcpkg_include_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_INC_DIR)
    include_directories(${VCPKG_INC_DIR})
endif()

add_executable(Maxminddb Main.cpp)
if(USE_CMAKE)
    #target_link_libraries(Maxminddb PRIVATE nlohmann_json nlohmann_json::nlohmann_json)
elseif(VCPKG_DIR)
    check_vcpkg_lib_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_LIB_DIR)
    check_vcpkg_maxminddb_lib("libmaxminddb" ${VCPKG_LIB_DIR} MAXMINDDB)
    target_link_libraries(Maxminddb PRIVATE ${MAXMINDDB}  ws2_32 Rpcrt4 shlwapi)
endif()
