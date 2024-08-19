cmake_minimum_required(VERSION 3.30.2)

set(CMAKE_CUDA_ARCHITECTURES 86)

project(MatVec LANGUAGES CXX CUDA)

# Add Boost configuration
find_package(Boost REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

add_subdirectory(lib/pybind11)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS src/*.cpp src/*.cu)

pybind11_add_module(MatVec SHARED ${SRC_FILES})

set_property(TARGET MatVec PROPERTY CUDA_ARCHITECTURES 86)

target_compile_options(MatVec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86>)

target_compile_options(MatVec PRIVATE
    $<$<CONFIG:Debug>:
        $<$<COMPILE_LANGUAGE:CXX>:
            $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>:-g -Wall -Wextra -Werror -pedantic>
            $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>:/W4 /WX /Zi /Od>
        >
        $<$<COMPILE_LANGUAGE:CUDA>:
            $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>:-g -G --shared --cudart=static>
            $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>:-g -G --shared --cudart=static>
        >
    >
    $<$<CONFIG:Release>:
        $<$<COMPILE_LANGUAGE:CXX>:
            $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>:-O3 -Wall -Wextra -Werror -pedantic>
            $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>:/W4 /WX /O2>
        >
        $<$<COMPILE_LANGUAGE:CUDA>:
            $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>:-O3 --cudart=static --shared>
            $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>:-O3 --cudart=static --shared>
        >
    >
)

set_target_properties(MatVec PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(MatVec PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(MatVec PUBLIC pybind11::module ${Boost_LIBRARIES})
set_target_properties(MatVec PROPERTIES CUDA_STANDARD 17)
set_target_properties(MatVec PROPERTIES CXX_STANDARD 17)
set_target_properties(MatVec PROPERTIES CUDA_LINK_LIBRARIES_KEYWORD "PUBLIC")
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set_target_properties(MatVec PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(MatVec PUBLIC include ${Boost_INCLUDE_DIRS})
include_directories(include ${Boost_INCLUDE_DIRS})

# For CUDA files
if(CUDA_FOUND)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -I${Boost_INCLUDE_DIRS})
endif()
