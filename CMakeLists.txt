cmake_minimum_required(VERSION 3.30.2)

# if(UNIX AND NOT APPLE)
#     set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# endif()

project(MatVec LANGUAGES CXX CUDA)

# Find pybind11
execute_process(
	COMMAND python3 -m pybind11 --cmakedir
	OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
find_package(pybind11 REQUIRED)

# Set CUDA architectures
# set(CUDA_ARCHITECTURES 86) # Adjust according to your target GPU architectures

# Recursively find all .cpp and .cu files in the src directory
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS src/*.cpp src/*.cu)

# Add library with CUDA support
add_library(MatVec SHARED ${SRC_FILES})

set_property(TARGET MatVec PROPERTY CUDA_ARCHITECTURES 86)

target_compile_options(MatVec PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86>
)

target_compile_options(MatVec PRIVATE
	# Debug options
	$<$<CONFIG:Debug>:
		$<$<COMPILE_LANGUAGE:CXX>:
			$<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>:-g -Wall -Wextra -Werror -pedantic>
			$<$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>:/Zi /W4 /WX>
		>
		$<$<COMPILE_LANGUAGE:CUDA>:
			$<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>:-g -G --shared --cudart=static>
			$<$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>:-g -G --shared --cudart=shared>
		>
	>

	# Release options
	$<$<CONFIG:Release>:
		$<$<COMPILE_LANGUAGE:CXX>:
			$<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>:-O3 -Wall -Wextra -Werror -pedantic>
			$<$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>:/O2 /W4 /WX>
		>
		$<$<COMPILE_LANGUAGE:CUDA>:
			$<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>:-O3 --cudart=shared --shared>
			$<$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>:-O3 --cudart=shared --shared>
		>
	>
)



# Ensure position-independent code is enabled
set_target_properties(MatVec PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Enable separable compilation if needed
set_target_properties(MatVec PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link against pybind11
target_link_libraries(MatVec PUBLIC pybind11::module)

# Optionally, specify the CUDA standard
set_target_properties(MatVec PROPERTIES CUDA_STANDARD 17)
set_target_properties(MatVec PROPERTIES CXX_STANDARD 17)

# Ensure shared libraries are used
set_target_properties(MatVec PROPERTIES CUDA_LINK_LIBRARIES_KEYWORD "PUBLIC")

# Remove shared library prefix if needed
set(CMAKE_SHARED_LIBRARY_PREFIX "")

# Additional linker flags to ensure shared libraries are used
set_target_properties(MatVec PROPERTIES LINKER_LANGUAGE CUDA)

target_include_directories(MatVec PUBLIC include)
include_directories(include)

