cmake_minimum_required(VERSION 3.0)
project(Datetime)

include_directories(${CMAKE_HOME_DIRECTORY})

if(USE_CMAKE)
    find_package(nlohmann_json CONFIG REQUIRED)
    include_directories(${Boost_INCLUDE_DIRS})    
elseif(VCPKG_DIR)
    #### cmake模块
    set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake" ${CMAKE_MODULE_PATH})
    include(Vcpkg)

    #### vcpkg 目录
    message("VCPKG_DIR: ${VCPKG_DIR}")

    #### 获取平台目标类型 x86, x64
    check_platform_size(PLATFORM_TARGET)
    check_vcpkg_include_dir(${PLATFORM_TARGET} ${VCPKG_DIR} VCPKG_INC_DIR)
    include_directories(${VCPKG_INC_DIR})
endif()

add_executable(NlohmannJson Main.cpp NlohmannJson.cpp)
if(USE_CMAKE)
    target_link_libraries(NlohmannJson PRIVATE nlohmann_json nlohmann_json::nlohmann_json)
elseif(VCPKG_DIR)
    #target_link_libraries(Datetime PRIVATE ${Boost_LIBRARIES})
endif()
