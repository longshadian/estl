cmake_minimum_required(VERSION 3.0)
set_property(GLOBAL PROPERTY USE_FOLDERS On)

#project(BlacklistPluginTest)
set(OUTPUT test_bin)

set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake" ${CMAKE_MODULE_PATH})

#### cmake模块
include(Functions)
include(Vcpkg)
include(VcpkgBoost)

#### 工程文件目录结构
set(SRC_DIR "${CMAKE_HOME_DIRECTORY}/test")
auto_sources(SRC_FILES "*.cpp" "RECURSE" "${SRC_DIR}")
auto_sources(SRC_HFILES "*.h" "RECURSE" "${SRC_DIR}")
auto_source_group(test ${SRC_DIR} ${SRC_FILES} ${SRC_HFILES})

#### 编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS false)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_IS_MSVC)
    add_definitions("/D _WIN32_WINNT=0x0601")
    add_definitions("/D _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS=1")
    add_definitions("/D _CRT_SECURE_NO_WARNINGS=1")
    add_definitions("/utf-8")     ### utf-8 no-bom warning C4819: 
    
elseif(CMAKE_CXX_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wconversion -Wshadow")	
endif()
message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

#### 获取平台目标类型 x86, x64
check_platform_size(PLATFORM_TARGET)
## message("platform_type: ${PLATFORM_TARGET}")

#set(VCPKG_DIR "c:/vcpkg")

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
add_executable(${OUTPUT} ${SRC_FILES} ${SRC_HFILES})
if(CMAKE_CXX_COMPILER_IS_MSVC)
    if (VCPKG_DIR)
        check_vcpkg_lib_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_LIB_DIR)
        check_vcpkg_boost_lib("boost_system" ${VCPKG_LIB_DIR} BOOST_SYSTEM)
        check_vcpkg_boost_lib("boost_filesystem" ${VCPKG_LIB_DIR} BOOST_FILESYSTEM)
    endif()

    target_link_libraries(${OUTPUT} ${BOOST_FILESYSTEM} ${BOOST_SYSTEM})
elseif(UNIX)
    target_link_libraries(${OUTPUT})
endif()


