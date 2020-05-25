cmake_minimum_required(VERSION 3.0)
project(Libzip)

if(USE_CMAKE)
    find_package(libzip REQUIRED)
elseif(VCPKG_DIR)
    #### cmake模块
    set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake" ${CMAKE_MODULE_PATH})
	include(Vcpkg)
	include(VcpkgLibzip)

    #### 获取平台目标类型 x86, x64
    check_platform_size(PLATFORM_TARGET)
    check_vcpkg_include_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_INC_DIR)
    include_directories(${VCPKG_INC_DIR})
endif()

if(WIN32)
	add_definitions("/std:c++17")
	add_definitions("/D _WIN32_WINNT=0x0601")
	add_definitions("/D _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS")
	add_definitions("/utf-8")     ### utf-8 no-bom warning C4819: 
endif(WIN32)

add_executable(Libzip Main.cpp Unzip.cpp Unzip.h)
if(USE_CMAKE)
    #target_link_libraries(Libzip PRIVATE ZLIB::ZLIB)
elseif(VCPKG_DIR)
	check_vcpkg_lib_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_LIB_DIR)
	check_vcpkg_libzip_lib("zip" ${VCPKG_LIB_DIR} VCPKG_LIBZIP_LIB)
	target_link_libraries(Libzip ${VCPKG_LIBZIP_LIB})
endif()

