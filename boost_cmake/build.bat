@ECHO off

cd /d %~dp0
"C:\Program Files\CMake\bin\cmake.exe" -DCMAKE_BUILD_TYPE=Debug -DUSE_VCPKG=1 -B "build"
 
pause 
