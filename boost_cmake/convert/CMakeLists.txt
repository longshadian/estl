cmake_minimum_required(VERSION 3.0)
project(Convert)

MESSAGE(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(Convert Convert.cpp)
target_link_libraries(Convert PRIVATE ${Boost_LIBRARIES})