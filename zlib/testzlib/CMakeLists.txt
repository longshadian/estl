cmake_minimum_required(VERSION 3.0)
project(TestZlib)

MESSAGE(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(ZLIB REQUIRED)
#include_directories(${Boost_INCLUDE_DIRS})

add_executable(TestZlib TestZlib.cpp)
target_link_libraries(TestZlib PRIVATE ZLIB::ZLIB)