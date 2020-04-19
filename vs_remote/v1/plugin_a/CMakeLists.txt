cmake_minimum_required(VERSION 3.0)
set_property(GLOBAL PROPERTY USE_FOLDERS On)

set(OUTPUT a)

set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake")

#### cmake模块
include(Functions)
include(Vcpkg)
include(VcpkgBoost)

#### 工程文件目录结构
set(SRC_DIR "${CMAKE_HOME_DIRECTORY}/plugin_a")
auto_sources(SRC_FILES "*.cpp" "RECURSE" "${SRC_DIR}")
auto_sources(SRC_HFILES "*.h" "RECURSE" "${SRC_DIR}")
auto_sources(SRC_HPPFILES "*.hpp" "RECURSE" "${SRC_DIR}")
auto_source_group(plugin_a ${SRC_DIR} ${SRC_FILES} ${SRC_HFILES} ${SRC_HPPFILES})

#### 编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_IS_MSVC)
    #add_definitions(/std:c++17)
    add_definitions("/D _WIN32_WINNT=0x0601")
    add_definitions("/D _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS")
    add_definitions("/D _CRT_SECURE_NO_WARNINGS")
    add_definitions("/utf-8")     ### utf-8 no-bom warning C4819: 
elseif(CMAKE_CXX_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g -Werror -Wall -Wextra -Wconversion -Wshadow -Wl,-Bsymbolic")	
endif()

#### 获取平台目标类型 x86, x64
check_platform_size(PLATFORM_TARGET)
## message("platform_type: ${PLATFORM_TARGET}")

#### 头文件
include_directories(${CMAKE_HOME_DIRECTORY})
if(CMAKE_CXX_COMPILER_IS_MSVC)
    if(VCPKG_DIR)
        check_vcpkg_include_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_INC_DIR)
        message("VCPKG_INC_DIR: ${VCPKG_INC_DIR}")
        include_directories(${VCPKG_INC_DIR})
    endif()
endif()

#### 编译成动态库
add_library(${OUTPUT} SHARED ${SRC_FILES} ${SRC_HFILES} ${SRC_HPPFILES})
set_property(TARGET ${OUTPUT} PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ${OUTPUT} PROPERTY PREFIX "")
if(CMAKE_CXX_COMPILER_IS_MSVC)
    check_vcpkg_lib_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_LIB_DIR)
    check_vcpkg_boost_lib("boost_system" ${VCPKG_LIB_DIR} BOOST_SYSTEM)

    target_link_libraries(${OUTPUT} ${BOOST_SYSTEM})
elseif(CMAKE_CXX_COMPILER_IS_GNUCXX)
    target_link_libraries(${OUTPUT} boost_system pthread)
endif()


