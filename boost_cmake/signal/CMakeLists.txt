cmake_minimum_required(VERSION 3.0)
project(Signal)

if(USE_CMAKE)
    find_package(Boost COMPONENTS system REQUIRED)
    include_directories(${Boost_INCLUDE_DIRS})    
elseif(VCPKG_DIR)
    #### cmake模块
    set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake" ${CMAKE_MODULE_PATH})
    include(Vcpkg)
    include(VcpkgBoost)
    include(Functions)

    #### 获取平台目标类型 x86, x64
    util_check_platform_size(PLATFORM_TARGET)
    ## message("platform_type: ${PLATFORM_TARGET}")
    vcpkg_find_include_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_INC_DIR)
    #message("VCPKG_INC_DIR: ${VCPKG_INC_DIR}")
    include_directories(${VCPKG_INC_DIR})
endif()

#### 编译选项
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS false)
set(CMAKE_CXX_STANDARD_REQUIRED true)
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_definitions("/D _WIN32_WINNT=0x0601")
	add_definitions("/D _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS")
	add_definitions("/utf-8")     ### utf-8 no-bom warning C4819: 
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")    
	set(CMAKE_CXX_FLAGS "-Wall ${CMAKE_CXX_FLAGS}")	
endif()

add_executable(Signal Signal.cpp)
if(USE_CMAKE)
elseif(USE_VCPKG)
    vcpkg_find_lib_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_LIB_DIR)
    vcpkg_find_boost("boost_system" ${VCPKG_LIB_DIR} BOOST_SYSTEM)

    set(Boost_LIBRARIES ${BOOST_SYSTEM})
endif()
target_link_libraries(Signal PRIVATE ${Boost_LIBRARIES})

