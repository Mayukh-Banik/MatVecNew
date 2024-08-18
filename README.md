# MatVec

**MatVec** is a python library meant to be a replacement for any NumPy operations on CUDA capable GPUs. Creation of MatVec classes currently can only be done with existing NumPy arrays due to too many creation routines for one person at this time, although once created, existing arrays will behave exactly the same as NumPy code on the CPU. 

Must be built from source with CUDA > 12 due to a bug in CUDA 11.5.

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

**Note**: The input array must be readable, contiguous in memory, and have a data type of `np.float64`.

## Installation Guide

### Linux

1. Update and upgrade your system:

    ```bash
    sudo apt update
    sudo apt upgrade -y
    ```

2. Install Python pip and pytest:

    ```bash
    sudo apt install python3-pip
    pip install pytest
    ```

3. Install CMake:

    ```bash
    sudo apt install snapd
    sudo snap install cmake --classic
    ```

4. Install CUDA:

    - Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) and follow the instructions for your system.
    - Ensure that CUDA is accessible to CMake. For Pop OS, you can install the NVIDIA CUDA Toolkit:

        ```bash
        sudo apt install nvidia-cuda-toolkit
        ```

        Note: The version in the repository may be outdated. Custom installation methods may be necessary, current version is 11.6, ensure CUDA is >= 12.

5. Install CMake:

    - Minimum version required: 3.30.2. Lower versions might work but are not tested.

6. Run ./BuildScipt.sh which will also run the tests

### Windows

1. Install Visual Studio, or build tools for Windows.

2. Install CMake, preferred with pip, but use any distribution as long as its added to path.

3. Install pytest
    '''bash
    pip install pytest
    '''

4. Install CUDA toolkit 12.5: https://developer.nvidia.com/cuda-12-5-0-download-archive?target_os=Windows

5. Rub BuildScript.bat. This will also run the test suite.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please contact [your email or contact information here].

