cmake_minimum_required(VERSION 3.0)
project(ProgramOptions)

MESSAGE(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(ProgramOptions ProgramOptions.cpp)
target_link_libraries(ProgramOptions PRIVATE ${Boost_LIBRARIES})