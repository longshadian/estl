@ECHO off

cd /d %~dp0
"C:\Program Files\CMake\bin\cmake.exe" -DCMAKE_BUILD_TYPE=Debug -DVCPKG_DIR="C:/vcpkg" -B "build"
 
pause 
