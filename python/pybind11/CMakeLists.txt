cmake_minimum_required(VERSION 3.0)


project("estl-Pybind11")
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/../../cmake")

add_subdirectory(src)
add_subdirectory(CppModule)

