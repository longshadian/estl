# CMakeLists.txt
cmake_minimum_required(VERSION 3.0)
SET(CMAKE_TOOLCHAIN_FILE C:\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake)
project(test)

MESSAGE(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(sqlite3 CONFIG REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE sqlite3)