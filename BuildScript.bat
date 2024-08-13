@echo off
setlocal

REM Check if an argument is provided
if "%1"=="" (
    echo Usage: %0 [debug^|release]
    exit /b 1
)

REM Set the build type based on the argument
set BUILD_TYPE=%1

REM Validate the argument
if /i not "%BUILD_TYPE%"=="debug" if /i not "%BUILD_TYPE%"=="release" (
    echo Invalid build type: %BUILD_TYPE%
    echo Usage: %0 [debug^|release]
    exit /b 1
)

REM Create the build directory if it doesn't exist
if not exist "build\%BUILD_TYPE%" (
    mkdir "build\%BUILD_TYPE%"
)

REM Configure and build the project
cmake -S . -B "build\%BUILD_TYPE%" -DCMAKE_BUILD_TYPE=%BUILD_TYPE%
cmake --build "build\%BUILD_TYPE%" --config %BUILD_TYPE%

endlocal
