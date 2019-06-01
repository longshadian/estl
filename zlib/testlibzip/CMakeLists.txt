cmake_minimum_required(VERSION 3.0)
project(TestLibzip)

MESSAGE(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(libzip REQUIRED)
#include_directories(${Boost_INCLUDE_DIRS})

add_executable(TestLibzip TestLibzip.cpp)
target_link_libraries(TestLibzip PRIVATE ZLIB::ZLIB)