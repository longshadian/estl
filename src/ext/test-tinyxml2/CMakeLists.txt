cmake_minimum_required(VERSION 3.0)
project(Tinyxml2)


if(USE_CMAKE)
	find_package(tinyxml2 CONFIG REQUIRED)
elseif(VCPKG_DIR)
    #### cmake模块
    set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake" ${CMAKE_MODULE_PATH})
	include(Vcpkg)
	include(VcpkgTinyxml2)

    #### 获取平台目标类型 x86, x64
    check_platform_size(PLATFORM_TARGET)
    check_vcpkg_include_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_INC_DIR)
    include_directories(${VCPKG_INC_DIR})
endif()

if(WIN32)
	ADD_DEFINITIONS("/std:c++17")
	ADD_DEFINITIONS("/D _WIN32_WINNT=0x0601")
	ADD_DEFINITIONS("/D _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS")
	ADD_DEFINITIONS("/utf-8")     ### utf-8 no-bom warning C4819: 
endif(WIN32)

add_executable(Tinyxml2 Main.cpp)

if(USE_CMAKE)
	target_link_libraries(Tinyxml2 PRIVATE tinyxml2::tinyxml2)
elseif(VCPKG_DIR)
	check_vcpkg_lib_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_LIB_DIR)
	check_vcpkg_tinyxml2_lib("tinyxml2" ${VCPKG_LIB_DIR} VCPKG_TINYXML2_LIB)
	target_link_libraries(Tinyxml2 ${VCPKG_TINYXML2_LIB})
endif()
