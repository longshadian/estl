cmake_minimum_required(VERSION 3.10)

set(EXT_OUTPUT ${CMAKE_BINARY_DIR}/ext_output)
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${EXT_OUTPUT}/debug/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${EXT_OUTPUT}/debug/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${EXT_OUTPUT}/debug/bin)

    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${EXT_OUTPUT}/release/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${EXT_OUTPUT}/release/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${EXT_OUTPUT}/release/lib)
elseif()
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${EXT_OUTPUT}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${EXT_OUTPUT}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${EXT_OUTPUT}/bin)
endif()
#[[
message("-------------------->CMAKE_SOURCE_DIR      ${CMAKE_SOURCE_DIR}")
message("-------------------->CMAKE_BINARY_DIR      ${CMAKE_BINARY_DIR}")
message("-------------------->CMAKE_HOME_DIRECTORY  ${CMAKE_HOME_DIRECTORY}")
message("-------------------->EXT_OUTPUT            ${EXT_OUTPUT}")
#]]

#add_subdirectory(test-enoki)
#add_subdirectory(test-fmt)
#add_subdirectory(test-geolite2)
#add_subdirectory(test-jsoncpp)
#add_subdirectory(test-libzip)
#add_subdirectory(test-nlohmann-json)
#add_subdirectory(test-spdlog)
#add_subdirectory(test-tinyxml2)
#add_subdirectory(test-zlib)
#add_subdirectory(test-easyloggingpp)
#add_subdirectory(test-zlib-other)
add_subdirectory(test-sqlitecpp)




