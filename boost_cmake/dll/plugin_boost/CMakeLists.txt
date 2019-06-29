cmake_minimum_required(VERSION 3.0)
project(PluginBoost)

MESSAGE(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
MESSAGE(STATUS "CMAKE_HOME_DIRECTORY: ${CMAKE_HOME_DIRECTORY}")

SET(OUTPUT_DIR ${CMAKE_BINARY_DIR}/output)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}/debug/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}/debug/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}/debug/bin)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}/release/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}/release/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}/release/lib)

find_package(Boost COMPONENTS filesystem system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

INCLUDE_DIRECTORIES(${CMAKE_HOME_DIRECTORY})

ADD_LIBRARY(PluginBoost SHARED PluginBoost.cpp)
target_link_libraries(PluginBoost PRIVATE ${Boost_LIBRARIES})