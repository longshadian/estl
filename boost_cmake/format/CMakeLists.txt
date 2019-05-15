cmake_minimum_required(VERSION 3.0)
project(Format)

MESSAGE(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(Format Format.cpp)
target_link_libraries(Format PRIVATE ${Boost_LIBRARIES})