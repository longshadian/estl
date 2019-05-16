cmake_minimum_required(VERSION 3.0)
project(HlohmannJson)

find_package(nlohmann_json CONFIG REQUIRED)
#include_directories(${Boost_INCLUDE_DIRS})

add_executable(HlohmannJson HlohmannJson.cpp)
target_link_libraries(HlohmannJson PRIVATE nlohmann_json nlohmann_json::nlohmann_json)