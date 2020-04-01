# CMakeLists.txt
cmake_minimum_required(VERSION 3.0)
SET(CMAKE_TOOLCHAIN_FILE C:\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake)
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS On)


PROJECT(tboost)
SET(OUTPUT tboost)
#SET(OUTPUT_DIR ${CMAKE_BINARY_DIR}/output)

#### 源码工程目录
FILE(GLOB SRC	*.cpp *.h)
FILE(GLOB SRC_NET              net/*.cpp net/*.h)
SOURCE_GROUP("" FILES ${SRC})
SOURCE_GROUP(TREE ${PROJECT_SOURCE_DIR} FILES ${SRC_NET})

INCLUDE_DIRECTORIES(${CMAKE_HOME_DIRECTORY}/tboost)

SET(SRC_LIST ${SRC}
	${SRC_NET}
    )


MESSAGE(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(sqlite3 CONFIG REQUIRED)

add_executable(${OUTPUT} ${SRC_LIST})
target_link_libraries(${OUTPUT} PRIVATE sqlite3)