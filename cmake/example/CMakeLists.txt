CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS On)

PROJECT(CMAKE_TEST)
SET(main_src Main.cpp)
SET(output main)

## 判断g++ vc++ 版本号，是否支持C++17
SET(MSVC_VERSION_MIN "1915")
#set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
IF(MSVC)
    IF(MSVC_VERSION STREQUAL ${MSVC_VERSION_MIN})
        #SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
        #ADD_DEFINITIONS("/std:c++17")
        MESSAGE(STATUS "set CMAKE_CXX_FLAGS /std:c++17")
    ELSE()
        MESSAGE(FATAL_ERROR "MSVC_VERSION: ${MSVC_VERSION} not support std=c++17."
                "use Visual Studio 2017 subversion:15.7.6")
    ENDIF()
ENDIF()

MESSAGE(STATUS ${PROJECT_SOURCE_DIR})

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
#INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/func)

#SET(src_all "" CACHE INTERNAL "ICD Library location")

FILE(GLOB src	*.cpp *.h)
SOURCE_GROUP("" FILES ${src})

FILE(GLOB src_func func/*.cpp func/*.h)
#SOURCE_GROUP(TREE ${PROJECT_SOURCE_DIR} PREFIX "src" FILES ${src_func})
SOURCE_GROUP(TREE ${PROJECT_SOURCE_DIR} FILES ${src_func})

FILE(GLOB src_func_ext func/ext/*.cpp func/ext/*.h)
#SOURCE_GROUP(TREE ${PROJECT_SOURCE_DIR} PREFIX "src" FILES ${src_func_ext})
SOURCE_GROUP(TREE ${PROJECT_SOURCE_DIR} FILES ${src_func_ext})

SET(src_all ${src} ${src_func} ${src_func_ext})

INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindBoost.cmake)

MESSAGE(FATAL_ERROR "FIND_BOOST: ${FIND_BOOST}")

#ADD_SUBDIRECTORY(func)

MESSAGE(FATAL_ERROR, "boost_libraries: ${BOOST_LIBRARIES}")

ADD_EXECUTABLE(${output} ${src_all})
#target_link_libraries(Main)