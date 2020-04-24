cmake_minimum_required(VERSION 3.0)

project("estl-RestSDK")
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/../cmake")

add_subdirectory(test-client)
add_subdirectory(test-server)

