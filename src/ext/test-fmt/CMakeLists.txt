cmake_minimum_required(VERSION 3.0)
project(Fmt)


if(USE_CMAKE)
	find_package(fmt CONFIG REQUIRED)
	include_directories(${fmt_INCLUDE_DIRS})
elseif(VCPKG_DIR)
    #### cmake模块
    #set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake" ${CMAKE_MODULE_PATH})
	include(Vcpkg)
	include(VcpkgFmt)

    #### 获取平台目标类型 x86, x64
    util_check_platform_size(PLATFORM_TARGET)
    vcpkg_find_include_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_INC_DIR)
    include_directories(${VCPKG_INC_DIR})
endif()

if(WIN32)
	ADD_DEFINITIONS("/std:c++17")
	ADD_DEFINITIONS("/D _WIN32_WINNT=0x0601")
	ADD_DEFINITIONS("/D _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS")
	ADD_DEFINITIONS("/utf-8")     ### utf-8 no-bom warning C4819: 
endif(WIN32)

add_executable(Fmt Fmt.cpp)

if(USE_CMAKE)
	target_link_libraries(Fmt PRIVATE fmt::fmt fmt::fmt-header-only)
elseif(VCPKG_DIR)
	vcpkg_find_lib_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_LIB_DIR)
	vcpkg_find_fmt("fmt" ${VCPKG_LIB_DIR} VCPKG_FMT_LIB)
	target_link_libraries(Fmt ${VCPKG_FMT_LIB})
endif()
