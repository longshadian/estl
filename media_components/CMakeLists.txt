cmake_minimum_required(VERSION 3.18)

project(media_components)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_subdirectory(common)
add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tests)


