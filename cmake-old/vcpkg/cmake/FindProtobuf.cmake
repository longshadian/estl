
FIND_PATH(PROTOBUF_INCLUDE_DIR google/protobuf/message.h
    PATHS
    /usr/include
    /usr/local/include
    E:\\gitpro\\vcpkg\\installed\\x86-windows\\include
    
    NO_DEFAULT_PATH
)
IF(NOT PROTOBUF_INCLUDE_DIR)
    MESSAGE(FATAL_ERROR "Can't find PROTOBUF_INCLUDE_DIR: ${PROTOBUF_INCLUDE_DIR}")
  ELSE()
    MESSAGE("Set PROTOBUF_INCLUDE_DIR: ${PROTOBUF_INCLUDE_DIR}")
ENDIF()


IF(WIN32)
    FIND_PATH(PROTOBUF_LIBRARY_DIR libprotobufd.lib
        PATHS
        E:\\gitpro\\vcpkg\\installed\\x86-windows\\debug\\lib
        
        NO_DEFAULT_PATH
    )
ELSEIF(UNIX)
    FIND_PATH(PROTOBUF_LIBRARY_DIR libprotobuf.so
        PATHS
        /usr/lib
        /usr/local/lib
        
        NO_DEFAULT_PATH
    )
ENDIF(WIN32)
IF(NOT PROTOBUF_LIBRARY_DIR)
    MESSAGE(FATAL_ERROR "Can't find PROTOBUF_LIBRARY_DIR: ${PROTOBUF_LIBRARY_DIR}")
  ELSE()
    MESSAGE("Set PROTOBUF_LIBRARY_DIR: ${PROTOBUF_LIBRARY_DIR}")
ENDIF()

IF(WIN32)
    FIND_LIBRARY(LIB_PROTOBUF libprotobufd.lib PATHS ${PROTOBUF_LIBRARY_DIR})
ELSEIF(UNIX)
    FIND_LIBRARY(LIB_PROTOBUF libprotobuf.so PATHS ${PROTOBUF_LIBRARY_DIR})
ENDIF(WIN32)

IF(NOT LIB_PROTOBUF)
    MESSAGE(FATAL_ERROR "Can't find LIB_PROTOBUF")
ENDIF()

SET(PROTOBUF_LIBRARIES ${LIB_PROTOBUF}
    CACHE STRING "protobuf libraries")

