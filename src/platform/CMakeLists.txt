cmake_minimum_required(VERSION 3.0)
set_property(GLOBAL PROPERTY USE_FOLDERS On)

project(TestPlatform)
set(OUTPUT TestPlatform)
set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/output)

if(WIN32)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}/debug/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}/debug/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}/debug/bin)

    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}/release/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}/release/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}/release/lib)
elseif(UNIX)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}/bin)
endif(WIN32)


#set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/../cmake")

include(Functions)

set(ROOT_DIR "${CMAKE_HOME_DIRECTORY}/platform")
auto_sources(SRC_FILES "*.cpp" "RECURSE" "${ROOT_DIR}")
auto_sources(SRC_HFILES "*.h" "RECURSE" "${ROOT_DIR}")
auto_source_group(${PROJECT_NAME} ${ROOT_DIR} ${SRC_FILES} ${SRC_HFILES})

include_directories(${CMAKE_HOME_DIRECTORY})


if(WIN32)
    add_definitions(/std:c++17)
    add_definitions("/D _WIN32_WINNT=0x0601")
    add_definitions("/D _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS")
    add_definitions("/utf-8")     ### utf-8 no-bom warning C4819: 
    ## 设置windows 窗口程序
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")

elseif(UNIX)
    add_definitions(-g -Wall -Wextra  -std=c++17)
endif()

add_executable(${OUTPUT} ${SRC_FILES} ${SRC_HFILES})
        

