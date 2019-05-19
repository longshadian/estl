用sqlite3 来测试vcpkg安装的库是否能被vs2017使用.
1.用powser shell build
	mkdir build
	cd build
	cmake .. "-DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake" -G "Visual Studio 15 2017 Win64"
	cmake --build .
	.\Debug\main.exe
	
2.用vs2017 open cmake 打开项目，需要在project()之前添加SET(CMAKE_TOOLCHAIN_FILE C:\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake)
 或者添加CMakeSettings.json,并添加以下内容
	"buildCommandArgs": "-v",
	"variables": [
        {
          "name": "CMAKE_TOOLCHAIN_FILE",
          "value": "C:\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake"
        }
      ]

3.用cmake gui创建项目时，需要在project()之前添加SET(CMAKE_TOOLCHAIN_FILE C:\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake)

	  
	  