cmake_minimum_required(VERSION 3.0)
set_property(GLOBAL PROPERTY USE_FOLDERS On)

project(Tech)
set(OUTPUT Tech)
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


set(CMAKE_MODULE_PATH
    #"${CMAKE_HOME_DIRECTORY}/cmake"
    ${CMAKE_MODULE_PATH}
    )

#include(Functions)

set(TECH_DIR "${CMAKE_HOME_DIRECTORY}/neo/tech")
auto_sources(SRC_FILES "*.cpp" "RECURSE" "${TECH_DIR}")
auto_sources(SRC_HFILES "*.h" "RECURSE" "${TECH_DIR}")
auto_source_group(tech ${TECH_DIR} ${SRC_FILES} ${SRC_HFILES})

#include_directories(${CMAKE_HOME_DIRECTORY}/neo)
include_directories(${CMAKE_HOME_DIRECTORY}/neo/tech)

if(WIN32)
    add_definitions(/std:c++17)
    add_definitions("/D _WIN32_WINNT=0x0601")
    add_definitions("/D _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS")
    add_definitions("/utf-8")     ### utf-8 no-bom warning C4819: 
    ## 设置windows 窗口程序
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:WINDOWS")

elseif(UNIX)
    #add_definitions(-g -Wall -Wextra -Werror -std=c++17)
    #add_definitions(-DBOOST_LOG_DYN_LINK)
    #set(THIRD_LIBRARIES ${THIRD_LIBRARIES} pthread)
    add_definitions(-g -Wall -Wextra  -std=c++17)
endif()

#### third libraries
set(THIRD_LIBRARIES 
    #${RDKAFKA_LIBRARIES}
    #${ZOOKEEPER_LIBRARIES}
    #${PROTOBUF_LIBRARIES}
    #${LIBCONFIG_LIBRARIES}
    )
#add_library(${OUTPUT} SHARED ${SRC_LIST})
add_executable(${OUTPUT} ${SRC_FILES} ${SRC_HFILES})
#target_link_libraries(${OUTPUT} PRIVATE fmt-header-only)
        

