

function(util_auto_sources RETURN_VALUE PATTERN SOURCE_SUBDIRS)
  if ("${SOURCE_SUBDIRS}" STREQUAL "RECURSE")
    SET(PATH ".")
    if (${ARGC} EQUAL 4)
      list(GET ARGV 3 PATH)
    endif ()
  endif()

  if ("${SOURCE_SUBDIRS}" STREQUAL "RECURSE")
    unset(${RETURN_VALUE})
    file(GLOB SUBDIR_FILES "${PATH}/${PATTERN}")
    list(APPEND ${RETURN_VALUE} ${SUBDIR_FILES})

    file(GLOB subdirs RELATIVE ${PATH} ${PATH}/*)

    foreach(DIR ${subdirs})
      if (IS_DIRECTORY ${PATH}/${DIR})
        if (NOT "${DIR}" STREQUAL "CMakeFiles")
          file(GLOB_RECURSE SUBDIR_FILES "${PATH}/${DIR}/${PATTERN}")
          list(APPEND ${RETURN_VALUE} ${SUBDIR_FILES})
        endif()
      endif()
    endforeach()
  else()
    file(GLOB ${RETURN_VALUE} "${PATTERN}")

    foreach (PATH ${SOURCE_SUBDIRS})
      file(GLOB SUBDIR_FILES "${PATH}/${PATTERN}")
      list(APPEND ${RETURN_VALUE} ${SUBDIR_FILES})
    endforeach()
  endif ()

  set(${RETURN_VALUE} ${${RETURN_VALUE}} PARENT_SCOPE)
endfunction(util_auto_sources)


# Automatically create source_group directives for the sources passed in.
function(util_auto_source_group rootName rootDir)
  file(TO_CMAKE_PATH "${rootDir}" rootDir)
  string(LENGTH "${rootDir}" rootDirLength)
  set(sourceGroups)
  foreach (fil ${ARGN})
    file(TO_CMAKE_PATH "${fil}" filePath)
    string(FIND "${filePath}" "/" rIdx REVERSE)
    if (rIdx EQUAL -1)
      message(FATAL_ERROR "Unable to locate the final forward slash in '${filePath}'!")
    endif()
    string(SUBSTRING "${filePath}" 0 ${rIdx} filePath)

    string(LENGTH "${filePath}" filePathLength)
    string(FIND "${filePath}" "${rootDir}" rIdx)
    if (rIdx EQUAL 0)
      math(EXPR filePathLength "${filePathLength} - ${rootDirLength}")
      string(SUBSTRING "${filePath}" ${rootDirLength} ${filePathLength} fileGroup)

      string(REPLACE "/" "\\" fileGroup "${fileGroup}")
      set(fileGroup "\\${rootName}${fileGroup}")

      list(FIND sourceGroups "${fileGroup}" rIdx)
      if (rIdx EQUAL -1)
        list(APPEND sourceGroups "${fileGroup}")
        source_group("${fileGroup}" REGULAR_EXPRESSION "${filePath}/[^/.]+.(c|cpp|cxx|cc|h|hpp|cu)$")
      endif()
    endif()
  endforeach()
endfunction(util_auto_source_group)


## 获取 x64 x86
function(util_check_platform_size out_ARGV)
    if(${CMAKE_SIZEOF_VOID_P} STREQUAL "4")
        set(${out_ARGV} "x86" PARENT_SCOPE)
    else()
        set(${out_ARGV} "x64" PARENT_SCOPE)
    endif()
endfunction(util_check_platform_size)

