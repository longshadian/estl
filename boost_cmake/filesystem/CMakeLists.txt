cmake_minimum_required(VERSION 3.0)
project(Filesystem)

MESSAGE(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(Boost COMPONENTS filesystem date_time REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(Filesystem Filesystem.cpp)
target_link_libraries(Filesystem PRIVATE ${Boost_LIBRARIES})