
SET(VCPKG_ROOT_DIR ${VCPKG_DIR})
#[[
IF(WIN32)
    SET(VCPKG_ROOT_DIR ${VCPKG_DIR}\\installed)
ELSEIF(UNIX)
    SET(VCPKG_ROOT_DIR ${VCPKG_DIR}/installed)
ENDIF(WIN32)
]]

FUNCTION(CHECK_PLATFORM_SIZE argv)
IF(${CMAKE_SIZEOF_VOID_P} STREQUAL "4")
    SET(${argv} "x86" PARENT_SCOPE)
ELSE()
    SET(${argv} "x64" PARENT_SCOPE)
ENDIF()
ENDFUNCTION()

#SET(PLATFORM_TARGET "")
CHECK_PLATFORM_SIZE(PLATFORM_TARGET)
MESSAGE(STATUS "PLATFORM_TARGET ${PLATFORM_TARGET}")

OPTION(VCPKG "use vcpkg " ON)
IF(VCPKG)
    FIND_PATH(VCPKG_DIR E:\\gitpro\\vcpkg\\installed
    #FIND_PATH(VCPKG_DIR
        PATHS
        C:\\
        NO_DEFAULT_PATH
    )

    SET(VCPKG_DIR E:\\gitpro\\vcpkg\\installed)
    #MESSAGE("root VCPKG_DIR: ${VCPKG_DIR}")

    IF(NOT VCPKG_DIR)
        MESSAGE(FATAL_ERROR "Can't find VCPKG_DIR: ${VCPKG_DIR}")
    ELSE()
        MESSAGE(STATUS "Set VCPKG_DIR: ${VCPKG_DIR}")
    ENDIF()
ENDIF()


IF(WIN32)
    IF(PLATFORM_TARGET STREQUAL "x86")
        SET(VCPKG_ROOT_DIR ${VCPKG_DIR}\\x86-windows)
    ELSE()
        SET(VCPKG_ROOT_DIR ${VCPKG_DIR}\\x64-windows)
    ENDIF()
    SET(VCPKG_ROOT_INCLUDE_DIR ${VCPKG_ROOT_DIR}\\include)
ELSEIF(UNIX)
    IF(PLATFORM_TARGET STREQUAL "x86")
        SET(VCPKG_ROOT_DIR ${VCPKG_DIR}/x86-windows)
    ELSE()
        SET(VCPKG_ROOT_DIR ${VCPKG_DIR}/x64-windows)
    ENDIF()
    SET(VCPKG_ROOT_INCLUDE_DIR ${VCPKG_ROOT_DIR}/include)
ENDIF(WIN32)

