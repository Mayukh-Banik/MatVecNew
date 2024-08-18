# MatVec

**MatVec** is a library for performing matrix-vector operations on the GPU using NumPy arrays, PyBind11, and CUDA. It is designed for NVIDIA GPUs.

## Features

- Leverages GPU acceleration for matrix-vector operations.
- Drop-in usage with existing NumPy arrays.

## Usage

To use MatVec, follow these steps:

1. Import the library and NumPy:

    ```python
    import MatVec as mv
    import numpy as np
    ```

2. Create a NumPy array and pass it to MatVec:

    ```python
    a = np.array(1)
    b = mv(a)
    ```

**Note**: The input array must be readable, contiguous in memory, and have a data type of `float64`.

## Installation Guide

### Linux

1. Update and upgrade your system:

    ```bash
    sudo apt update
    sudo apt upgrade -y
    ```

2. Install Python pip:

    ```bash
    sudo apt install python3-pip
    ```

3. Install PyBind11:

    ```bash
    pip install pybind11
    ```

4. Install Snapd and CMake:

    ```bash
    sudo apt install snapd
    sudo snap install cmake --classic
    ```

5. Install CUDA:

    - Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) and follow the instructions for your system.
    - Ensure that CUDA is accessible to CMake. For Pop OS, you can install the NVIDIA CUDA Toolkit:

        ```bash
        sudo apt install nvidia-cuda-toolkit
        ```

        Note: The version in the repository may be outdated. Custom installation methods may be necessary.

6. Install CMake:

    - Minimum version required: 3.30.2. Lower versions might work but are not tested.

### Windows

1. Install CUDA and PyBind11.
2. Additional instructions and testing are forthcoming.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please contact [your email or contact information here].

