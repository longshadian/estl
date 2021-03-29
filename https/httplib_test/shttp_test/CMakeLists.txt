set(PROJECT_NAME shttp_test)

#### 编译选项
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
	set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wconversion -Wshadow -std=c++14 ${CMAKE_CXX_FLAGS}")	
endif()

include_directories(
    ${CMAKE_HOME_DIRECTORY}/httplib
    ${CMAKE_HOME_DIRECTORY}
)

#### 生成可执行文件
add_executable(${PROJECT_NAME} 
    Main.cpp SHttpServer.cpp)

#### 链接选项
target_link_libraries(${PROJECT_NAME} 
httplib
)

