@ECHO off

::echo %WINDIR%
::echo %AppData%

cd /d %~dp0
"C:\Program Files\CMake\bin\cmake.exe" -DCMAKE_BUILD_TYPE=Release -DVCPKG_DIR="C:/vcpkg" -B "build-release"
 
pause 
