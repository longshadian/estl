cmake_minimum_required(VERSION 3.0)

project(estl-src)

set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/../cmake")

add_subdirectory(platform)
add_subdirectory(dynamic)
add_subdirectory(bytebuffer)

