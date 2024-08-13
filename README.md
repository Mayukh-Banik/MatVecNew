**MatVec**

NumPy on the GPU made using PyBind11 and CUDA software. Only for NVIDIA GPUs at the moment.

This is meant to be drop in usage with NumPy arrays. 

Usage:
import MatVec as mv
import numpy as np
a = np.array(1)
b = mv(a)

Can only be called from existing NumPy arrays that are readable, contiguos in memory, and have data types of float 64.


Installation guide:
Linux:
sudo apt update
sudo apt upgrade -y
Install pip
pip install pybind11
Install CUDA:
  Ensure that CUDA is accessible to CMake, I used Pop OS sudo apt install nvidia-cuda-toolkit
  The version is slightly outdated, custom methods IDK how to do
Install CMake: 
  min version is 3.30.2, but it can be manually lowered if desired, haven't tested with anything lower though.
Windows:
  Install CUDA, PyBind11. Will test later.
