cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME Easylogpp)
project(${PROJECT_NAME})


if(USE_CMAKE)
	#find_package(fmt CONFIG REQUIRED)
	#include_directories(${fmt_INCLUDE_DIRS})
elseif(VCPKG_DIR)
    #### cmake模块
    #set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake" ${CMAKE_MODULE_PATH})
	include(Vcpkg)
	include(Functions)

    #### 获取平台目标类型 x86, x64
    util_check_platform_size(PLATFORM_TARGET)
    #vcpkg_find_include_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_INC_DIR)
    include_directories(${VCPKG_INC_DIR})
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	add_definitions("/std:c++17")
	add_definitions("/D _WIN32_WINNT=0x0601")
	add_definitions("/D _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS")
	add_definitions("/utf-8")     ### utf-8 no-bom warning C4819: 
	
	add_definitions("/D ELPP_THREAD_SAFE")
	add_definitions("/D ELPP_FORCE_USE_STD_THREAD")
	add_definitions("/D ELPP_DISABLE_FATAL_LOGS")
	add_definitions("/D ELPP_NO_DEFAULT_LOG_FILE")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
	add_definitions(-DELPP_THREAD_SAFE)
	add_definitions(-DELPP_FORCE_USE_STD_THREAD)
	add_definitions(-DELPP_DISABLE_FATAL_LOGS)
	add_definitions(-DELPP_NO_DEFAULT_LOG_FILE)

	set(CMAKE_CXX_FLAGS "-std=c++17" ${CMAKE_CXX_FLAGS})
endif()


add_executable(${PROJECT_NAME} 
	main.cpp easylogging++.h easylogging++.cc
	Easylogging.h Easylogging.cpp
	)

target_link_libraries(${PROJECT_NAME} pthread)

