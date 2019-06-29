cmake_minimum_required(VERSION 3.0)
project(zflog)

MESSAGE(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

#find_package(Boost REQUIRED)
#include_directories(${Boost_INCLUDE_DIRS})

add_executable(Test Main.cpp Log.cpp Log.h)
#target_link_libraries(Test PRIVATE ${Boost_LIBRARIES})