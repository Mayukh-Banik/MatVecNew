cmake_minimum_required(VERSION 3.30.2)

project(MatVec LANGUAGES CXX CUDA)

execute_process(
	COMMAND python3 -m pybind11 --cmakedir
	OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
find_package(pybind11 REQUIRED)
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS src/*.cpp src/*.cu)
add_library(MatVec SHARED ${SRC_FILES})
set_property(TARGET MatVec PROPERTY CUDA_ARCHITECTURES 86)
target_compile_options(MatVec PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86>
)
target_compile_options(MatVec PRIVATE
	$<$<CONFIG:Debug>:
		$<$<COMPILE_LANGUAGE:CXX>:
			$<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>:-g -Wall -Wextra -Werror -pedantic>
		>
		$<$<COMPILE_LANGUAGE:CUDA>:
			$<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>:-g -G --shared --cudart=static>
		>
	>
	$<$<CONFIG:Release>:
		$<$<COMPILE_LANGUAGE:CXX>:
			$<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>:-O3 -Wall -Wextra -Werror -pedantic>
		>
		$<$<COMPILE_LANGUAGE:CUDA>:
			$<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>:-O3 --cudart=shared --shared>
		>
	>
)
set_target_properties(MatVec PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(MatVec PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(MatVec PUBLIC pybind11::module)
set_target_properties(MatVec PROPERTIES CUDA_STANDARD 17)
set_target_properties(MatVec PROPERTIES CXX_STANDARD 17)
set_target_properties(MatVec PROPERTIES CUDA_LINK_LIBRARIES_KEYWORD "PUBLIC")
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set_target_properties(MatVec PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(MatVec PUBLIC include)
include_directories(include)
