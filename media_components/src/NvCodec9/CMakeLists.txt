set_property(GLOBAL PROPERTY USE_FOLDERS On)

#project(NvCodec9 CXX)
set(PROJECT_NAME NvCodec9)

if(NOT CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-10.2")
endif()
find_package(CUDA REQUIRED)

#[[
message("-------> CUDA_INCLUDE_DIRS         ${CUDA_INCLUDE_DIRS}")
message("-------> CUDA_LIBRARIES            ${CUDA_LIBRARIES}")
message("-------> CUDA_CUFFT_LIBRARIES      ${CUDA_CUFFT_LIBRARIES}")
message("-------> CUDA_CUBLAS_LIBRARIES     ${CUDA_CUBLAS_LIBRARIES}")
message("-------> CUDA_cupti_LIBRARY        ${CUDA_cupti_LIBRARY}")
message("-------> CUDA_curand_LIBRARY       ${CUDA_curand_LIBRARY}")
message("-------> CUDA_cusparse_LIBRARY     ${CUDA_cusparse_LIBRARY}")
message("-------> CUDA_npp_LIBRARY          ${CUDA_npp_LIBRARY}")
message("-------> CUDA_nppc_LIBRARY         ${CUDA_nppc_LIBRARY}")
message("-------> CUDA_nppi_LIBRARY         ${CUDA_nppi_LIBRARY}")
message("-------> CUDA_npps_LIBRARY         ${CUDA_npps_LIBRARY}")
message("-------> CUDA_nvcuvenc_LIBRARY     ${CUDA_nvcuvenc_LIBRARY}")
message("-------> CUDA_nvcuvid_LIBRARY      ${CUDA_nvcuvid_LIBRARY}")
#]]

#### cmake module路径
set(CMAKE_MODULE_PATH
    "${CMAKE_HOME_DIRECTORY}/cmake"
    "${CMAKE_MODULE_PATH}"
)

#### cmake模块
include(Functions)

#### 工程文件目录结构
set(SRC_DIR "${CMAKE_HOME_DIRECTORY}/src/NvCodec9")
util_auto_sources(SRC_C_FILES "*.c" "RECURSE" "${SRC_DIR}")
util_auto_sources(SRC_CPP_FILES "*.cpp" "RECURSE" "${SRC_DIR}")
util_auto_sources(SRC_CU_FILES "*.cu" "RECURSE" "${SRC_DIR}")
util_auto_sources(SRC_H_FILES "*.h" "RECURSE" "${SRC_DIR}")
util_auto_sources(SRC_HPP_FILES "*.hpp" "RECURSE" "${SRC_DIR}")
util_auto_source_group(src ${SRC_DIR} 
    ${SRC_C_FILES}
    ${SRC_CPP_FILES}
    ${SRC_CU_FILES}
    ${SRC_H_FILES}
    ${SRC_HPP_FILES}
)

#### 编译选项
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
	set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")	
endif()

#### 获取平台目标类型 x86, x64
util_check_platform_size(PLATFORM_TARGET)
message("platform_type: ${PLATFORM_TARGET}")

include_directories(
    "${CMAKE_HOME_DIRECTORY}/src/NvCodec9/"
    "${CMAKE_HOME_DIRECTORY}/src/NvCodec9/include"
#"${CMAKE_HOME_DIRECTORY}/src"
#  "/home/bolan/libs/include"
)

set(CUDA_NVCC_FLAGS "-std=c++14")
#### 生成可执行文件
#link_directories("/home/bolan/libs/lib")
#CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${SRC_C_FILES} ${SRC_CPP_FILES} ${SRC_CU_FILES})
CUDA_ADD_LIBRARY(${PROJECT_NAME} SHARED ${SRC_C_FILES} ${SRC_CPP_FILES} ${SRC_CU_FILES})
#### 链接选项
#target_link_libraries(${PROJECT_NAME} nvcuvid nvidia-encode cuda)

    