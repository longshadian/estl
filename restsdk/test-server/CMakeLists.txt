cmake_minimum_required(VERSION 3.0)
project(test-server)

if(USE_CMAKE)
    find_package(cpprestsdk REQUIRED)
    #include_directories(${Boost_INCLUDE_DIRS})    
elseif(VCPKG_DIR)
    #### cmake模块
    set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake" ${CMAKE_MODULE_PATH})
    include(Vcpkg)
    include(VcpkgCppRest)

    #### 获取平台目标类型 x86, x64
    check_platform_size(PLATFORM_TARGET)
    ## message("platform_type: ${PLATFORM_TARGET}")
    check_vcpkg_include_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_INC_DIR)
    #message("VCPKG_INC_DIR: ${VCPKG_INC_DIR}")
    include_directories(${VCPKG_INC_DIR})
endif()


add_executable(server Server.cpp)
if(USE_CMAKE)
    target_link_libraries(server PRIVATE cpprestsdk::cpprest)
elseif(VCPKG_DIR)
    check_vcpkg_lib_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_LIB_DIR)
    check_vcpkg_cpprest_lib("cpprest_2_10" ${VCPKG_LIB_DIR} CPPREST)
    message("CPPREST ${CPPREST}")
    target_link_libraries(server PRIVATE ${CPPREST})
else()
    include_directories("/home/cgy/Desktop/github/cpprestsdk/Release/include")

#[[
    link_directories("/home/cgy/Desktop/github/cpprestsdk/build/Release/Binaries")
    foreach (dir ${dirs})
        message(STATUS " link_directories    '${dir}'")
    endforeach ()
#]]
    set(CMAKE_PREFIX_PATH "/home/cgy/Desktop/github/cpprestsdk/build/Release/Binaries")
    message("==========> CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}")
##[[
    find_library(CPPREST
          NAMES cpprest
          #PATH_SUFFIXES lib
          #PATHS
            "home/cgy/Desktop/github/cpprestsdk/build/Release/Binaries"
        )
    message("==============> ${CPPREST}")
#]]

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17")

    message("==============> CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}")

    target_link_libraries(server ${CPPREST} boost_system crypto ssl)
    #target_link_libraries(server cpprest boost_system crypto ssl)
endif()

