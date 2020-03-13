cmake_minimum_required(VERSION 3.0)
set_property(GLOBAL PROPERTY USE_FOLDERS On)

project(CppModule)
set(OUTPUT CppModule)
set(OUTPUT_DIR ${CMAKE_BINARY_DIR})

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


set(CMAKE_MODULE_PATH
    "${CMAKE_HOME_DIRECTORY}/cmake"
    ${CMAKE_MODULE_PATH}
    )

    SET(LIBRARY_OUTPUT_NAME "CppModule.pyd")

include(Functions)

set(ROOT_DIR "${CMAKE_HOME_DIRECTORY}/CppModule")
auto_sources(SRC_FILES "*.cpp" "RECURSE" "${ROOT_DIR}")
auto_sources(SRC_HFILES "*.h" "RECURSE" "${ROOT_DIR}")
auto_source_group("" ${ROOT_DIR} ${SRC_FILES} ${SRC_HFILES})

include_directories("C:/vcpkg/installed/x64-windows/include")
include_directories("C:/vcpkg/installed/x64-windows/include/python3.7")
include_directories(${CMAKE_HOME_DIRECTORY}/CppModule)

if(WIN32)
    add_definitions(/std:c++17)
    add_definitions("/D _WIN32_WINNT=0x0601")
    add_definitions("/D _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS")
    add_definitions("/utf-8")     ### utf-8 no-bom warning C4819: 
elseif(UNIX)
    #add_definitions(-g -Wall -Wextra -Werror -std=c++17)
    #add_definitions(-DBOOST_LOG_DYN_LINK)
    add_definitions(-g -Wall -Wextra  -std=c++17)
endif()

#### third libraries
set(THIRD_LIBRARIES 
    "C:/vcpkg/installed/x64-windows/lib/python37.lib"
    )

add_library(${OUTPUT} SHARED ${SRC_FILES} ${SRC_HFILES})
#add_library(CppModule.pyd ALIAS ${OUTPUT})
#SET_TARGET_PROPERTIES (${OUTPUT} PROPERTIES LIBRARY_OUTPUT_NAME_RELEASE "CppModule.pyd")
#set(CMAKE_RELEASE_POSTFIX _d)
#set_target_properties (${OUTPUT} PROPERTIES RELEASE_POSTFIX _d)
#message(STATUS "${OUTPUT} ${LIBRARY_OUTPUT_NAME}   sss")        
target_link_libraries(${OUTPUT} ${THIRD_LIBRARIES})



