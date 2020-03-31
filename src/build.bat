@ECHO off

::echo %WINDIR%
::echo %AppData%

cd /d %~dp0
"C:\Program Files\CMake\bin\cmake.exe" -DCMAKE_BUILD_TYPE=Debug -B "build"
 
pause 
