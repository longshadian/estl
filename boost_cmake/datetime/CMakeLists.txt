cmake_minimum_required(VERSION 3.0)
project(Datetime)

MESSAGE(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(Boost COMPONENTS date_time REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(Datetime Datetime.cpp)
target_link_libraries(Datetime PRIVATE ${Boost_LIBRARIES})