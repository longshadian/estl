cmake_minimum_required(VERSION 3.10)

project("estl-Test")
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/../cmake")

add_subdirectory(test-fmt)
add_subdirectory(test-nlohmann-json)
add_subdirectory(test-geolite2)
add_subdirectory(test-tinyxml2)
add_subdirectory(test-zlib)
add_subdirectory(test-libzip)

