cmake_minimum_required(VERSION 3.0)
project(Download)
SET(OUTPUT Download)
SET(OUTPUT_DIR ${CMAKE_BINARY_DIR}/output)

##[
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}/debug/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}/debug/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}/debug/bin)

SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}/release/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}/release/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}/release/lib)
##]

MESSAGE(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

FILE(GLOB DOWNLOAD	./*.cpp ./*.h)
SOURCE_GROUP("" FILES ${DOWNLOAD})

SET(SRC_LIST ${DOWNLOAD} 
        )

### 依赖库
find_package(boost COMPONENTS filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIRS})

find_package(nlohmann_json CONFIG REQUIRED)
#target_link_libraries(main PRIVATE nlohmann_json nlohmann_json::nlohmann_json)

find_package(OpenSSL REQUIRED)

#### 编译选项
IF(WIN32)
	ADD_DEFINITIONS(/std:c++17)
	ADD_DEFINITIONS("/D _WIN32_WINNT=0x0601")
	ADD_DEFINITIONS("/D _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS")
	ADD_DEFINITIONS("/utf-8")     ### utf-8 no-bom warning C4819: 
ELSEIF(UNIX)
	#ADD_DEFINITIONS(-g -Wall -Wextra -Werror -std=c++17)
	#ADD_DEFINITIONS(-DBOOST_LOG_DYN_LINK)
	ADD_DEFINITIONS(-g -Wall -Wextra  -std=c++17)
ENDIF()

add_executable(${OUTPUT} ${SRC_LIST})
target_link_libraries(${OUTPUT} PRIVATE 
	${Boost_LIBRARIES} 
	${CURL_LIBRARIES}
	nlohmann_json nlohmann_json::nlohmann_json
	OpenSSL::SSL OpenSSL::Crypto
	)