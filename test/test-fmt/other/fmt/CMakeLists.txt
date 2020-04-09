cmake_minimum_required(VERSION 3.0)
project(Fmt)

MESSAGE(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(fmt CONFIG REQUIRED)
include_directories(${fmt_INCLUDE_DIRS})

if(WIN32)
	ADD_DEFINITIONS(/std:c++17)
	ADD_DEFINITIONS("/D _WIN32_WINNT=0x0601")
	ADD_DEFINITIONS("/D _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS")
	ADD_DEFINITIONS("/utf-8")     ### utf-8 no-bom warning C4819: 
ENDIF(WIN32)

add_executable(Fmt Fmt.cpp)
target_link_libraries(Fmt PRIVATE fmt::fmt fmt::fmt-header-only)
