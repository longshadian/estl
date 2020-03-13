@ECHO off

::echo %WINDIR%
::echo %AppData%

cd /d %~dp0
"C:\Program Files\CMake\bin\cmake.exe" -S "D:\temp\pybind11test" -B "D:\temp\pybind11test\build"
 
pause 
