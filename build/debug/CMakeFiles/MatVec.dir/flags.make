# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# compile CUDA with /usr/bin/nvcc
# compile CXX with /usr/bin/c++
CUDA_DEFINES = -DMatVec_EXPORTS

CUDA_INCLUDES = --options-file CMakeFiles/MatVec.dir/includes_CUDA.rsp

CUDA_FLAGS = -g "--generate-code=arch=compute_52,code=[compute_52,sm_52]" -Xcompiler=-fPIC -arch=sm_86 -g -G --shared --cudart=shared

CXX_DEFINES = -DMatVec_EXPORTS

CXX_INCLUDES = -I/home/weket/Documents/Code/include -isystem /home/weket/.local/lib/python3.10/site-packages/pybind11/include -isystem /usr/include/python3.10

CXX_FLAGS = -g -fPIC -g -Wall -Wextra -Werror -pedantic

