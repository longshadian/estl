cmake_minimum_required(VERSION 3.0)
project(Blog)

MESSAGE(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(Boost COMPONENTS system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(Blog Main.cpp Logger.cpp)
target_link_libraries(Blog PRIVATE ${Boost_LIBRARIES})