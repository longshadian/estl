cmake_minimum_required(VERSION 3.0)
project(BufferStream)

MESSAGE(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(Boost COMPONENTS system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(BufferStream BufferStream.cpp)
target_link_libraries(BufferStream PRIVATE ${Boost_LIBRARIES})