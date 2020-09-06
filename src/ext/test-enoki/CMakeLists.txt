cmake_minimum_required(VERSION 3.0)
project(Enoki)


include_directories("D:/github/enoki/include")

if(USE_CMAKE)
	
elseif(VCPKG_DIR)
    #### cmake模块
    #set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake" ${CMAKE_MODULE_PATH})
	include(Functions)
	include(Vcpkg)
	

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

add_executable(Enoki Enoki.cpp)

if(USE_CMAKE)
	target_link_libraries(Enoki PRIVATE)
elseif(VCPKG_DIR)

	target_link_libraries(Enoki ${VCPKG_FMT_LIB})
endif()
