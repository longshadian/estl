cmake_minimum_required(VERSION 3.0)
project(Interpress)

MESSAGE(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(Interpress Interpress.cpp)
target_link_libraries(Interpress PRIVATE ${Boost_LIBRARIES})