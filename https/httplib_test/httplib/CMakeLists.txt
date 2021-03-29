set(PROJECT_NAME httplib)
project(${PROJECT_NAME} CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS false)
set(CMAKE_CXX_STANDARD_REQUIRED true)

#set(CMAKE_CXX_FLAGS "")

add_library(${PROJECT_NAME} httplib.cpp)

