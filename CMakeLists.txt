cmake_minimum_required(VERSION 3.30.2)
project(MatVec LANGUAGES CXX CUDA)

# Find the pybind11 CMake directory using Python
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Add the pybind11 CMake directory to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})

# Find pybind11 package
find_package(pybind11 REQUIRED)

# Create a shared library
add_library(MatVec SHARED example.cpp)

set(CMAKE_SHARED_LIBRARY_PREFIX "")


# Link the pybind11 library
target_link_libraries(MatVec PUBLIC pybind11::module)


