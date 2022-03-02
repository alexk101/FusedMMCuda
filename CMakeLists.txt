cmake_minimum_required(VERSION 3.21)
project(fusedmm_cuda CUDA)
find_package(CUDAToolkit)
if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "Could not find CUDA toolkit")
endif()
if (NOT DEFINED $ENV{CUDA_HOME})
    set($ENV{CUDA_HOME} ${CUDAToolkit_LIBRARY_ROOT})
endif ()

set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
set(CMAKE_CUDA_STANDARD 20)
set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -lineinfo -gencode arch=compute_75,code=sm_75 --ptxas-options=-v --use_fast_math
)

add_executable(fusedmm_cuda sddmm_spmm_csr.cu)
target_compile_options(fusedmm_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
include_directories(${CUDA_TOOLKIT_INCLUDE_DIRECTORIES} common)

set_target_properties(fusedmm_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)